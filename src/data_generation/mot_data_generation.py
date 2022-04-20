import numpy as np
import itertools
from modules.realistic_radar_model.realistic_meas_model import RealisticMeasModel


def get_radar_measurement(obj):
    """
    Creates a measurement of (range, range_rate, azimuth angle) from an object, without any measurement noise.
    @param obj:
    @return:
    """
    range = np.linalg.norm(obj.pos)
    range_rate = np.dot(obj.pos / np.linalg.norm(obj.pos), obj.vel)
    azimuth = np.arctan2(obj.pos[1], obj.pos[0])
    return np.array([range, range_rate, azimuth])


class FieldOfView:
    def __init__(self, min_range, max_range, max_range_rate, min_theta, max_theta):
        self.min_range = min_range
        self.max_range = max_range
        self.min_range_rate = -max_range_rate
        self.max_range_rate = max_range_rate
        self.min_theta = min_theta
        self.max_theta = max_theta

    def __contains__(self, measurement):
        if not (self.min_range <= measurement[0] <= self.max_range):
            return False
        elif not (self.min_range_rate <= measurement[1] <= self.max_range_rate):
            return False
        elif not (self.min_theta <= measurement[2] <= self.max_theta):
            return False
        else:
            return True

    def area(self):
        range_length = self.max_range - self.min_range
        range_rate_length = self.max_range_rate - self.min_range_rate
        theta_length = self.max_theta - self.min_theta
        return range_length * range_rate_length * theta_length


class Object:

    def __init__(self, pos, vel, t, delta_t, sigma, id):
        self.pos = pos
        self.vel = vel
        self.delta_t = delta_t
        self.sigma = sigma
        self.state_history = np.array([np.concatenate([pos, vel, np.array([t])])])
        self.process_noise_matrix = sigma*np.array([[delta_t ** 3 / 3, delta_t ** 2 / 2], [delta_t ** 2 / 2, delta_t]])

        # Unique identifier for every object
        self.id = id

    def update(self, t, rng):
        """
        Updates this object's state using a discretized constant velocity model.
        """

        # Update position and velocity of the object in each dimension separately
        assert len(self.pos) == len(self.vel)
        process_noise = rng.multivariate_normal([0, 0], self.process_noise_matrix, size=len(self.pos))
        self.pos += self.delta_t * self.vel + process_noise[:,0]
        self.vel += process_noise[:,1]

        # Add current state to previous states
        current_state = np.concatenate([self.pos.copy(), self.vel.copy(), np.array([t])])
        self.state_history = np.vstack((self.state_history, current_state))

    def __repr__(self):
        return 'id: {}, pos: {}, vel: {}'.format(self.id, self.pos, self.vel)


class MotDataGenerator:
    def __init__(self, args, rng):
        if not (isinstance(args.data_generation.measurement_noise_stds, list) and
                len(args.data_generation.measurement_noise_stds)==3):
            raise ValueError(f"Specified measurement noise should be a list with three elements, got "
                             f"'{args.data_generation.measurement_noise_stds}' instead")

        self.start_pos_params = [args.data_generation.birth_process.mean_pos, args.data_generation.birth_process.cov_pos]
        self.start_vel_params = [args.data_generation.birth_process.mean_vel, args.data_generation.birth_process.cov_vel]
        self.prob_add_obj = args.data_generation.p_add
        self.prob_remove_obj = args.data_generation.p_remove
        self.delta_t = args.data_generation.dt
        self.process_noise_variance = args.data_generation.process_noise_variance
        self.prob_measure = args.data_generation.p_meas
        self.measurement_noise_stds = args.data_generation.measurement_noise_stds
        self.n_average_false_measurements = args.data_generation.n_avg_false_measurements
        self.n_average_starting_objects = args.data_generation.n_avg_starting_objects
        field_of_view_min_theta = args.data_generation.field_of_view.min_theta if args.data_generation.field_of_view.min_theta is not None else -np.pi
        field_of_view_max_theta = args.data_generation.field_of_view.max_theta if args.data_generation.field_of_view.max_theta is not None else np.pi
        self.field_of_view = FieldOfView(args.data_generation.field_of_view.min_range,
                                         args.data_generation.field_of_view.max_range,
                                         args.data_generation.field_of_view.max_range_rate,
                                         field_of_view_min_theta,
                                         field_of_view_max_theta)
        self.max_objects = args.data_generation.max_objects
        self.use_realistic_meas_noise = args.data_generation.use_realistic_meas_noise
        if f'get_{args.data_generation.prediction_target}_from_state' in globals():
            self.prediction_target = args.data_generation.prediction_target
        else:
            raise NotImplementedError(f'The chosen function for mapping state to ground-truth was no implemented: {args.data_generation.prediction_target}')
        self.rng = rng
        self.debug = False

        if self.use_realistic_meas_noise:
            self.realistic_meas_computer = RealisticMeasModel()

            # Make sure we're using hyperparameters compatible to the ones used to precompute the covariance matrices of
            # the realistic measurement model
            for s_y in args.data_generation.measurement_noise_stds:
                assert s_y is None, f"When using realistic measurement model sigma_y must be [None, None, None], but " \
                                    f"got {args.data_generation.measurement_noise_stds} instead."
            assert args.data_generation.field_of_view.min_range >= self.realistic_meas_computer.min_r
            assert args.data_generation.field_of_view.max_range <= self.realistic_meas_computer.max_r
            assert args.data_generation.field_of_view.min_theta >= self.realistic_meas_computer.min_theta
            assert args.data_generation.field_of_view.max_theta <= self.realistic_meas_computer.max_theta
        else:
            for s_y in args.data_generation.measurement_noise_stds:
                assert s_y is not None, f"Measurement noise cannot be None. Got {args.data_generation.measurement_noise_stds}."

        assert self.n_average_starting_objects != 0, 'Datagen does not currently work with n_avg_starting_objects equal to zero.'

        self.t = None
        self.objects = None
        self.trajectories = None
        self.measurements = None
        self.unique_ids = None
        self.unique_id_counter = None
        self.reset()

    def reset(self):
        self.t = 0
        self.objects = []
        self.trajectories = {}
        self.measurements = np.array([])
        self.unique_ids = np.array([], dtype='int64')
        self.unique_id_counter = itertools.count()

        # Add initial set of objects (re-sample until we get a nonzero value)
        n_starting_objects = 0
        while n_starting_objects == 0:
            n_starting_objects = self.rng.poisson(self.n_average_starting_objects)
        self.add_objects(n_starting_objects)

        # Measure the initial set of objects
        self.generate_measurements()

        if self.debug:
            print(n_starting_objects, 'starting objects')

    def create_new_object(self, pos, vel):
        return Object(pos=pos,
                      vel=vel,
                      t=self.t,
                      delta_t=self.delta_t,
                      sigma=self.process_noise_variance,
                      id=next(self.unique_id_counter))

    def create_n_objects(self, n):
        """
        Create `n` objects according to Gaussian birth model. Objects outside measurement FOV are discarded.
        """
        positions = self.rng.multivariate_normal(self.start_pos_params[0], self.start_pos_params[1], size=(n,))
        velocities = self.rng.multivariate_normal(self.start_vel_params[0], self.start_vel_params[1], size=(n,))
        objects = []
        for pos, vel in zip(positions, velocities):
            obj = self.create_new_object(pos, vel)
            if get_radar_measurement(obj) in self.field_of_view:
                objects.append(obj)
        return objects

    def add_objects(self, n):
        """
        Adds `n` new objects to `objects` list.
        """
        # Never add more objects than the maximum number of allowed objects
        n = min(n, self.max_objects-len(self.objects))
        if n == 0:
            return

        # Create new objects and save them in the datagen
        new_objects = self.create_n_objects(n)
        self.objects += new_objects

    def remove_far_away_objects(self):

        if len(self.objects) == 0:
            return

        # Check which objects left the FOV
        meas_coordinates_of_objects = np.array([get_radar_measurement(obj) for obj in self.objects])
        deaths = [meas_obj not in self.field_of_view for meas_obj in meas_coordinates_of_objects]

        # Save state history of objects that will be removed in self.trajectories
        for obj, death in zip(self.objects, deaths):
            if death:
                self.trajectories[obj.id] = obj.state_history[:-1]

        # Remove objects that left the measurement FOV
        self.objects = [o for o, r in zip(self.objects, deaths) if not r]

    def remove_objects(self, p):
        """
        Removes each of the objects with probability `p`.
        """

        # Compute which objects are removed in this time-step
        deaths = self.rng.binomial(n=1, p=p, size=len(self.objects))

        n_deaths = sum(deaths)
        if self.debug and (n_deaths > 0):
            print(n_deaths, 'objects were removed')

        # Save the trajectories of the removed objects
        for obj, death in zip(self.objects, deaths):
            if death:
                self.trajectories[obj.id] = obj.state_history

        # Remove them from the object list
        self.objects = [o for o, d in zip(self.objects, deaths) if not d]

    def get_prob_death(self, obj):
        return self.prob_remove_obj

    def remove_object(self, obj, p=None):
        """
        Removes an object based on its state
        """
        if p is None:
            p = self.get_prob_death(obj)

        r = self.rng.rand()

        if r < p:
            return True
        else:
            return False

    def generate_measurements(self):
        """
        Generates all measurements (true and false) for the current time-step.
        """

        # Decide which of the objects will be measured
        is_measured = self.rng.binomial(n=1, p=self.prob_measure, size=len(self.objects))
        measured_objects = [obj for obj, is_measured in zip(self.objects, is_measured) if is_measured]

        # Generate the true measurements' noise
        true_measurements = []
        if self.use_realistic_meas_noise:
            positions = np.array([get_radar_measurement(obj) for obj in measured_objects])
            measurement_covs = self.realistic_meas_computer.compute_covariance(positions)
            dim_measurements = len(self.measurement_noise_stds)
            measurement_noises = np.array([self.rng.multivariate_normal(np.zeros(dim_measurements), cov) for cov in measurement_covs])
        else:
            measurement_noises = self.rng.normal(0, self.measurement_noise_stds, size=(len(measured_objects), 3))

        # Generate true measurements (making sure they're inside the FOV)
        for i, obj in enumerate(measured_objects):
            m = get_radar_measurement(obj)
            measurement_with_time = np.append(m + measurement_noises[i, :], self.t)
            if measurement_with_time[:-1] in self.field_of_view:
                true_measurements.append(measurement_with_time)
        true_measurements = np.array(true_measurements)

        unique_obj_ids_true = [obj.id for obj in measured_objects]

        # Generate false measurements (uniformly distributed over measurement FOV)
        n_false_measurements = self.rng.poisson(self.n_average_false_measurements)
        false_measurements = \
            self.rng.uniform(low=[self.field_of_view.min_range, -self.field_of_view.max_range_rate, self.field_of_view.min_theta],
                             high=[self.field_of_view.max_range, self.field_of_view.max_range_rate, self.field_of_view.max_theta],
                             size=(n_false_measurements, 3))

        # Add time to false measurements
        times = np.repeat([[self.t]], n_false_measurements, axis=0)
        false_measurements = np.concatenate([false_measurements, times], 1)

        # Also save from which object each measurement came from (for contrastive learning later); -1 is for false meas.
        unique_obj_ids_false = [-1]*len(false_measurements)
        unique_obj_ids = np.array(unique_obj_ids_true + unique_obj_ids_false)

        # Concatenate true and false measurements in a single array
        if true_measurements.shape[0] and false_measurements.shape[0]:
            new_measurements = np.vstack([true_measurements, false_measurements])
        elif true_measurements.shape[0]:
            new_measurements = true_measurements
        elif false_measurements.shape[0]:
            new_measurements = false_measurements
        else:
            return

        # Shuffle all generated measurements and corresponding unique ids in unison
        random_idxs = self.rng.permutation(len(new_measurements))
        new_measurements = new_measurements[random_idxs]
        unique_obj_ids = unique_obj_ids[random_idxs]

        # Save measurements and unique ids
        self.measurements = np.vstack([self.measurements, new_measurements]) if self.measurements.shape[0] else new_measurements
        self.unique_ids = np.hstack([self.unique_ids, unique_obj_ids])

    def step(self, add_new_objects=True):
        """
        Performs one step of the simulation.
        """
        self.t += self.delta_t

        # Update the remaining ones
        for obj in self.objects:
            obj.update(self.t, self.rng)

        # Remove objects that left the field-of-view
        self.remove_far_away_objects()

        # Add new objects
        if add_new_objects:
            n_new_objs = self.rng.poisson(self.prob_add_obj)
            self.add_objects(n_new_objs)

        # Remove some of the objects
        self.remove_objects(self.prob_remove_obj)
        
        # Generate measurements
        self.generate_measurements()
        
        if self.debug:
            if n_new_objs > 0:
                print(n_new_objs, 'objects were added')
            print(len(self.objects))

    def finish(self):
        """
        Should be called after the last call to `self.step()`. Removes the remaining objects, consequently adding the
        remaining parts of their trajectories to `self.trajectories`.
        """
        self.remove_objects(1.0)

        label_data = []
        unique_label_ids = []

        # -1 is applied because we count t=0 as one time-step
        last_timestep = round(self.t / self.delta_t)
        for traj_id in self.trajectories:
            traj = self.trajectories[traj_id]
            last_state = traj[-1]
            if round(last_state[4] / self.delta_t) == last_timestep:  # last state of trajectory, time
                pos = globals()[f'get_{self.prediction_target}_from_state'](last_state)
                label_data.append(pos)
                unique_label_ids.append(traj_id)

        training_data = np.array(self.measurements.copy())
        unique_measurements_ids = self.unique_ids.copy()

        return training_data, label_data, unique_measurements_ids, unique_label_ids


def get_position_and_velocity_from_state(state):
    return state[:4].copy()
