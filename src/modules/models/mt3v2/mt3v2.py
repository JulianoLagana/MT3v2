import torch
from torch import nn
from modules.position_encoder import LearnedPositionEncoder
from modules.mlp import MLP
from modules.models.mt3v2.transformer import TransformerEncoder, TransformerDecoder, PreProccessor, TransformerEncoderLayer, TransformerDecoderLayer
from modules.contrastive_classifier import ContrastiveClassifier
from util.misc import NestedTensor, Prediction
import copy
import math


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MT3V2(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.d_detections = params.arch.d_detections
        self.temporal_encoder = LearnedPositionEncoder(params.data_generation.n_timesteps, params.arch.d_model)

        # Normalization factor to make all measurement dimensions have similar standard deviations
        self.measurement_normalization_factor = \
            torch.tensor(
                [params.data_generation.field_of_view.max_range - params.data_generation.field_of_view.min_range,
                 params.data_generation.field_of_view.max_range_rate,
                 params.data_generation.field_of_view.max_theta - params.data_generation.field_of_view.min_theta], device=torch.device(params.training.device))

        # Rescaling factor to map xy position of measurements to (0.25, 0.75) when creating the proposals for two-stage
        # decoder
        self.fov_rescaling_factor = params.data_generation.field_of_view.max_range * 4

        # Rescaling factor used to scale predictions made by the net (from -0.5 to 0.5) to the correct range in
        # state-space.
        self.output_scaling_factor = [params.data_generation.field_of_view.max_range * 4,
                                      params.data_generation.field_of_view.max_range_rate * 4]

        self.preprocessor = PreProccessor(params.arch.d_model,
                                          params.arch.d_detections,
                                          normalization_constant=self.measurement_normalization_factor)
        self.false_detect_embedding = nn.Embedding(1,params.arch.d_model) if params.arch.false_detect_embedding else None
        encoder_layer = TransformerEncoderLayer(params.arch.d_model,
                                                nhead=params.arch.encoder.n_heads,
                                                dim_feedforward=params.arch.encoder.dim_feedforward,
                                                dropout=params.arch.encoder.dropout,
                                                activation="relu",
                                                normalize_before=False,
                                                false_detect_embedding=self.false_detect_embedding)
        self.encoder = TransformerEncoder(encoder_layer, num_layers=params.arch.encoder.n_layers, norm=None)
        decoder_layer = TransformerDecoderLayer(params.arch.d_model,
                                                nhead=params.arch.decoder.n_heads,
                                                dim_feedforward=params.arch.decoder.dim_feedforward,
                                                dropout=params.arch.decoder.dropout,
                                                activation="relu",
                                                normalize_before=False)
        decoder_norm = nn.LayerNorm(params.arch.d_model)
        self.decoder = TransformerDecoder(decoder_layer,
                                          num_layers=params.arch.decoder.n_layers,
                                          norm=decoder_norm,
                                          with_state_refine=params.arch.with_state_refine)

        self.query_embed = nn.Embedding(params.arch.num_queries, params.arch.d_model)

        # Create pos/vel delta predictor and existence probability predictor
        self.prediction_space_dimensions = 2  # (x, y) position and velocity
        self.pos_vel_predictor = MLP(params.arch.d_model,
                                     hidden_dim=params.arch.d_prediction_hidden,
                                     output_dim=self.prediction_space_dimensions*2,
                                     num_layers=params.arch.n_prediction_layers)
        self.uncertainty_predictor = MLP(params.arch.d_model,
                                         hidden_dim=params.arch.d_prediction_hidden,
                                         output_dim=self.prediction_space_dimensions*2,
                                         num_layers=params.arch.n_prediction_layers,
                                         softplus_at_end=True)
        self.obj_classifier = nn.Linear(params.arch.d_model, 1)

        self.return_intermediate = params.loss.return_intermediate
        if self.params.loss.contrastive_classifier:
            self.contrastive_classifier = ContrastiveClassifier(params.arch.d_model)
        if self.params.loss.false_classifier:
            self.false_classifier = MLP(params.arch.d_model,
                                        hidden_dim=params.arch.d_prediction_hidden,
                                        output_dim=1,
                                        num_layers=1)
        self.two_stage = params.arch.two_stage
        self.d_model = params.arch.d_model

        self._reset_parameters()

        # Initialize delta predictions to zero
        nn.init.constant_(self.pos_vel_predictor.layers[-1].weight.data, 0)
        nn.init.constant_(self.pos_vel_predictor.layers[-1].bias.data, 0)

        # Clone prediction heads for all layers of the decoder (+1 for encoder if two-stage)
        num_pred = (self.decoder.num_layers + 1) if self.two_stage else self.decoder.num_layers
        self.obj_classifier = _get_clones(self.obj_classifier, num_pred)
        self.pos_vel_predictor = _get_clones(self.pos_vel_predictor, num_pred)
        self.uncertainty_predictor = _get_clones(self.uncertainty_predictor, num_pred)
        self.decoder.pos_vel_predictor = self.pos_vel_predictor
        self.decoder.uncertainty_predictor = self.uncertainty_predictor
        self.decoder.obj_classifier = self.obj_classifier

        if self.two_stage:
            # hack implementation for two-stage
            self.enc_output = nn.Linear(params.arch.d_model, params.arch.d_model)
            self.enc_output_norm = nn.LayerNorm(params.arch.d_model)

            self.pos_trans = nn.Linear(self.d_model, self.d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(self.d_model * 2)

            self.num_queries = params.arch.num_queries
        else:
            self.reference_points_linear = nn.Linear(params.arch.d_model, self.prediction_space_dimensions*2)
            nn.init.xavier_uniform_(self.reference_points_linear.weight.data, gain=1.0)
            nn.init.constant_(self.reference_points_linear.bias.data, 0.)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def gen_encoder_output_proposals(self, embeddings, memory_padding_mask, normalized_measurements):
        # Compute presigmoid version of normalized measurements
        normalized_measurements_presigmoid = torch.log(normalized_measurements / (1 - normalized_measurements))

        # Set to inf invalid measurements (masked or outside the FOV)
        output_proposals_valid = ((normalized_measurements > 0.01) & (normalized_measurements < 0.99)).all(-1,keepdim=True)
        normalized_measurements_presigmoid = normalized_measurements_presigmoid.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        normalized_measurements_presigmoid = normalized_measurements_presigmoid.masked_fill(~output_proposals_valid, float('inf'))

        # Mask embeddings of measurements that are actually just padding
        masked_embeddings = embeddings
        masked_embeddings = masked_embeddings.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        masked_embeddings = masked_embeddings.masked_fill(~output_proposals_valid, float(0))

        # Project embeddings
        projected_embeddings = self.enc_output_norm(self.enc_output(masked_embeddings))
        return projected_embeddings, normalized_measurements_presigmoid

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = self.d_model
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 2
        proposals = proposals.sigmoid() * scale
        # N, L, 2, num_pos_feats
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 2, num_pos_feats/2, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        # N, L, num_pos_feats*2
        return pos

    def get_two_stage_proposals(self, measurement_batch, embeddings):
        """
        Given a batch of measurements and their corresponding embeddings (computed by the encoder), this generates the
        object queries to be fed by the decoder, using the selection mechanism as explained in https://arxiv.org/abs/2104.00734

        @param measurement_batch: Batch of measurements, including their masks.
        @param embeddings: Embeddings computed by the encoder for each of the measurements.
        @return:
            object_queries: queries to be fed to the decoder.
            query_positional_encodings: positional encodings to be added to the object queries.
            reference_points: 2D position estimates to be used as starting points for iterative refinement in the
                decoder.
            enc_outputs_class: predicted existence probability for each measurement.
            enc_outputs_state: predicted adjustment delta for each measurement (measurements are adjusted by summing
                their corresponding deltas before using them as starting points for iterative refinement.
            enc_outputs_coord_unact: adjusted measurements using their corresponding predicted deltas.
        """
        n_measurements, _, c = embeddings.shape
        measurements = measurement_batch.tensors[:, :, :self.d_detections]

        # Compute xy position of the measurements using range and azimuth
        xs = measurements[:, :, 0] * (measurements[:, :, 2].cos())
        ys = measurements[:, :, 0] * (measurements[:, :, 2].sin())
        xy_measurements = torch.stack([xs, ys], 2)

        # Normalize measurements to 0.25 - 0.75 (to avoid extreme regions of the sigmoid)
        normalized_xy_meas = xy_measurements / self.fov_rescaling_factor + 0.5

        # Compute projected encoder memory + presigmoid normalized measurements (filtered using the masks)
        result = self.gen_encoder_output_proposals(embeddings.permute(1, 0, 2), measurement_batch.mask, normalized_xy_meas)
        projected_embeddings, normalized_meas_presigmoid = result

        # Compute scores and adjustments
        scores = self.decoder.obj_classifier[self.decoder.num_layers](projected_embeddings)
        scores = scores.masked_fill(measurement_batch.mask.unsqueeze(-1), -100_000_000)  # Set masked predictions to "0" probability
        adjustments = self.pos_vel_predictor[self.decoder.num_layers](projected_embeddings)

        # Concatenate initial velocity estimates to the measurements
        init_vel_estimates_presigmoid = torch.zeros_like(normalized_meas_presigmoid)
        normalized_meas_presigmoid = torch.cat((normalized_meas_presigmoid, init_vel_estimates_presigmoid,), dim=2)

        # Adjust measurements
        adjusted_normalized_meas_presigmoid = normalized_meas_presigmoid + adjustments
        adjusted_normalized_meas = adjusted_normalized_meas_presigmoid.sigmoid()

        # Select top-k scoring measurements and their corresponding embeddings
        topk_scores_indices = torch.topk(scores[..., 0], self.num_queries, dim=1)[1]
        repeated_indices = topk_scores_indices.unsqueeze(-1).repeat((1, 1, adjusted_normalized_meas_presigmoid.shape[2]))
        topk_adjusted_normalized_meas_presigmoid = torch.gather(adjusted_normalized_meas_presigmoid,
                                                                1,
                                                                repeated_indices).detach()
        topk_adjusted_normalized_meas = topk_adjusted_normalized_meas_presigmoid.sigmoid().permute(1, 0, 2)
        topk_memory = torch.gather(projected_embeddings.detach(), 1, topk_scores_indices.unsqueeze(-1).repeat(1,1,self.params.arch.d_model))

        # Compute object queries and their positional encodings by feeding the top-k memory through FFN+LayerNorm
        pos_trans_out = self.pos_trans_norm(self.pos_trans(topk_memory))
        query_positional_encodings, object_queries = torch.split(pos_trans_out, c, dim=2)
        query_positional_encodings = query_positional_encodings.permute(1, 0, 2)
        object_queries = object_queries.permute(1, 0, 2)

        return object_queries, query_positional_encodings, topk_adjusted_normalized_meas, scores, adjustments, adjusted_normalized_meas

    def forward(self, measurements: NestedTensor):
        mapped_time_idx = torch.round(measurements.tensors[:, :, -1] / self.params.data_generation.dt)
        time_encoding = self.temporal_encoder(mapped_time_idx.long())
        preprocessed_measurements = self.preprocessor(measurements.tensors[:, :, :self.d_detections])
        mask = measurements.mask

        batch_size, num_batch_max_meas, d_detections = preprocessed_measurements.shape
        preprocessed_measurements = preprocessed_measurements.permute(1, 0, 2)
        time_encoding = time_encoding.permute(1, 0, 2)

        # Feed measurements through encoder
        embeddings = self.encoder(preprocessed_measurements, src_key_padding_mask=mask, pos=time_encoding)

        # Compute optional classifications
        aux_classifications = {}
        if self.params.loss.contrastive_classifier:
            contrastive_classifications = self.contrastive_classifier(embeddings.permute(1, 0, 2), padding_mask=mask)
            aux_classifications['contrastive_classifications'] = contrastive_classifications
        if self.params.loss.false_classifier:
            false_classifications = self.false_classifier(embeddings)
            aux_classifications['false_classifications'] = false_classifications

        # Compute object queries for the decoder
        if self.two_stage:
            (object_queries,
             query_positional_encodings,
             topk_adjusted_normalized_meas,
             scores,
             adjustments,
             adjusted_normalized_meas) = self.get_two_stage_proposals(measurements, embeddings)
        else:
            query_positional_encodings = self.query_embed.weight.unsqueeze(1).repeat(1, batch_size, 1)
            object_queries = torch.zeros_like(query_positional_encodings)
            topk_adjusted_normalized_meas = self.reference_points_linear(query_positional_encodings).sigmoid()

        # Feed embeddings and object queries to decoder
        result = self.decoder(object_queries, embeddings,
                              encoder_embeddings_padding_mask=mask,
                              encoder_embeddings_positional_encoding=time_encoding,
                              object_queries_positional_encoding=query_positional_encodings,
                              reference_points=topk_adjusted_normalized_meas)
        intermediate_state_predictions_normalized, intermediate_uncertainties, intermediate_logits, debug_dict = result

        # Un-normalize state predictions
        intermediate_state_predictions = intermediate_state_predictions_normalized - 0.5
        intermediate_state_predictions[:, :, :, :2] *= self.output_scaling_factor[0]
        intermediate_state_predictions[:, :, :, 2:] *= self.output_scaling_factor[1]

        # Un-normalize encoder state predictions and make sure padded measurements cannot be matched / are far away
        if self.two_stage:
            adjusted_meas = adjusted_normalized_meas - 0.5
            adjusted_meas[:, :, :2] *= self.output_scaling_factor[0]
            adjusted_meas[:, :, 2:] *= self.output_scaling_factor[1]
            # Hack to make padded measurements never be matched
            adjusted_meas = adjusted_meas.masked_fill(mask.unsqueeze(-1).repeat(1,1,adjusted_meas.shape[-1]), self.output_scaling_factor[0]*5)

        # Pack output using standardized Prediction class
        prediction = Prediction(positions=intermediate_state_predictions[-1][:, :, :self.prediction_space_dimensions],
                                velocities=intermediate_state_predictions[-1][:, :, self.prediction_space_dimensions:],
                                uncertainties=intermediate_uncertainties[-1],
                                logits=intermediate_logits[-1])
        intermediate_predictions = [Prediction(positions=p[:, :, :self.prediction_space_dimensions],
                                               velocities=p[:, :, self.prediction_space_dimensions:],
                                               uncertainties=u,
                                               logits=l) for p, l, u in zip(intermediate_state_predictions[:-1],
                                                                            intermediate_logits[:-1],
                                                                            intermediate_uncertainties[:-1])] if self.return_intermediate else None
        encoder_prediction = Prediction(positions=adjusted_meas[:, :, :self.prediction_space_dimensions],
                                        velocities=adjusted_meas[:, :, self.prediction_space_dimensions:],
                                        logits=scores) if self.two_stage else None
        return prediction, intermediate_predictions, encoder_prediction, aux_classifications, debug_dict

    def to(self, device):
        super().to(device)
        if self.params.loss.contrastive_classifier:
            self.contrastive_classifier.to(device)

