import torch 
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.modules import ModuleList
from util.misc import inverse_sigmoid
from modules.transformer import PreProccessor, TransformerEncoderLayer, TransformerDecoderLayer, TransformerEncoder 
import copy
import math
from typing import Optional, List


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, false_detect_embedding=None):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self.false_detect_embedding = false_detect_embedding

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        if self.false_detect_embedding:
            n, bs, dim = q.shape
            nf, _ = self.false_detect_embedding.weight.shape
            false_detect_embedding = self.false_detect_embedding.weight.unsqueeze(-1).permute(0,2,1).repeat(1,bs,1)
            k = torch.cat((k,false_detect_embedding), dim=0)
            val = torch.cat((src, false_detect_embedding), dim=0)
            false_detect_mask = torch.zeros(bs,nf).bool().to(src_key_padding_mask.device)
            src_key_padding_mask = torch.cat((src_key_padding_mask, false_detect_mask), dim=1)
        else:
            val = src
        
        src2 = self.self_attn(q, k, value=val, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        if self.false_detect_embedding:
            n, bs, dim = q.shape
            nf, _ = self.false_detect_embedding.weight.shape
            false_detect_embedding = self.false_detect_embedding.weight.unsqueeze(-1).permute(0,2,1).repeat(1,bs,1)
            k = torch.cat((k,false_detect_embedding), dim=0)
            val = torch.cat((src2, false_detect_embedding), dim=0)
            false_detect_mask = torch.zeros(bs,nf).bool().to(src_key_padding_mask.device)
            src_key_padding_mask = torch.cat((src_key_padding_mask, false_detect_mask), dim=1)
        else:
            val = src
        
        src2 = self.self_attn(q, k, value=val, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, false_detect_embedding=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        if false_detect_embedding:
            for layer in self.layers:
                layer.false_detect_embedding = false_detect_embedding
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, debug=False, with_state_refine=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.position_predictor = None
        self.obj_classifier = None
        self.with_state_refine = with_state_refine

        self.debug = debug

    def forward(self,
                object_queries,
                encoder_embeddings,
                encoder_embeddings_padding_mask: Optional[Tensor] = None,
                encoder_embeddings_positional_encoding: Optional[Tensor] = None,
                object_queries_positional_encoding: Optional[Tensor] = None,
                reference_points: Optional[Tensor] = None):
        """
        Computes forward propagation through the decoder, taking as input object queries and the embeddings computed by
        the decoder.

        @param object_queries: Object queries, learned vectors used as input to the decoder.
        @param encoder_embeddings: Embeddings computed in the encoder.
        @param encoder_embeddings_padding_mask: Mask signaling which embeddings correspond to pad measurements, added
            to ensure homogeneity in the measurement vector.
        @param encoder_embeddings_positional_encoding: Positional encoding added to the encoder embeddings.
        @param object_queries_positional_encoding: Positional encoding added to the object queries.
        @param reference_points: Starting points used by iterative refinement.
        @return:
            intermediate_state_predictions_normalized: State predictions computed by each of the decoder layers,
                normalized to be between 0 and 1 (or 0.25 - 0.75?)
            intermediate_logits: Logits of the existence probabilities computed by each of the decoder layers.
            debug_dict: Dictionary possibly containing useful debug information, if self.debug is True.
        """

        intermediate_state_predictions_normalized = []
        intermediate_uncertainties = []
        intermediate_logits = []
        debug_dict = {'intermediate_attention': []}

        # Go through each layer of the decoder, producing estimates and performing iterative refinement
        for layer_idx, layer in enumerate(self.layers):

            # Compute embeddings at the current decoder layer
            object_queries, attn_maps = layer(object_queries, encoder_embeddings,
                                              memory_key_padding_mask=encoder_embeddings_padding_mask,
                                              pos=encoder_embeddings_positional_encoding,
                                              query_pos=object_queries_positional_encoding)

            # Compute probability of existence using the embeddings
            predicted_logits = self.obj_classifier[layer_idx](self.norm(object_queries).permute(1, 0, 2))

            # Compute deltas using the embeddings, sum them in pre-sigmoid space
            deltas = self.pos_vel_predictor[layer_idx](self.norm(object_queries))
            normalized_predicted_state_at_current_layer = (deltas + inverse_sigmoid(reference_points)).sigmoid()

            # Compute uncertainty using the embeddings
            uncertainties = self.uncertainty_predictor[layer_idx](self.norm(object_queries))

            # If using state refinement, update the reference point using the new prediction, but detached, so as to
            # break gradient flow from here
            if self.with_state_refine:
                reference_points = normalized_predicted_state_at_current_layer.detach()

            # Save intermediate results for auxiliary loss
            intermediate_state_predictions_normalized.append(normalized_predicted_state_at_current_layer.permute(1, 0, 2))
            intermediate_uncertainties.append(uncertainties.permute(1, 0, 2))
            intermediate_logits.append(predicted_logits)

            # Save debug information if needed
            if self.debug:
                debug_dict['intermediate_attention'].append(attn_maps)


        intermediate_state_predictions_normalized = torch.stack(intermediate_state_predictions_normalized)
        intermediate_uncertainties = torch.stack(intermediate_uncertainties)
        intermediate_logits = torch.stack(intermediate_logits)
        return intermediate_state_predictions_normalized, intermediate_uncertainties, intermediate_logits, debug_dict


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):

    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError("activation should be relu/gelu, not {}".format(activation))
