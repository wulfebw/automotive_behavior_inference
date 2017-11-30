using AutoRisk

function analyze_risk_dataset(output_filepath)
    dataset = h5open(output_filepath)
    features = read(dataset["risk/features"])
    targets = read(dataset["risk/targets"])
    println("avg features: $(mean(features, (2,3)))")
    avg_targets = mean(clamp(sum(targets, 2), 0, 1), 3)
    println("avg targets: $(avg_targets)")
    println("size of dataset features: $(size(features))")
    println("size of dataset targets: $(size(targets))")
    if exists(dataset, "risk/weights")
        weights = read(dataset["risk/weights"])
        inds = find(weights .!= 1.)

        if length(inds) > 0
                avg_prop_weight = mean(weights[1, inds])
                println("avg proposal weight: $(avg_prop_weight)")
                med_prop_weight = median(weights[1, inds])
                println("median proposal weight: $(med_prop_weight)")
        end
    end
end

function build_dataset_collector(
        output_filepath::String, 
        col_id::Int = 0,
        num_scenarios = 1;
        num_lanes = 1,
        min_num_veh = 30,
        max_num_veh = 30,
        min_base_speed = 12.,
        max_base_speed = 12.,
        min_vehicle_length = 3.5,
        max_vehicle_length = 3.5,
        min_vehicle_width = 1.5,
        max_vehicle_width = 1.5,
        min_init_dist = 10.,
        lon_accel_std_dev = 0.5,
        lat_accel_std_dev = 0.1,
        overall_response_time = 0.,
        err_p_a_to_i = 0.,
        err_p_i_to_a = 1.,
        sampling_time = .1,
        sampling_period = .1,
        num_runs = 1,
        veh_idx_can_change = false,
        feature_step_size = 5,
        feature_timesteps = 50,
        chunk_dim = 1,
        roadway_length = 120.,
        roadway_radius = 30.,
        burn_in_time = 60.
    )

    # roadway generator
    roadway = gen_stadium_roadway(num_lanes, length = roadway_length, radius = roadway_radius)
    roadway_gen = StaticRoadwayGenerator(roadway)

    # scene gen
    scene = Scene(max_num_veh)
    scene_gen = HeuristicSceneGenerator(
        min_num_veh, 
        max_num_veh, 
        min_base_speed,
        max_base_speed,
        min_vehicle_length,
        max_vehicle_length,
        min_vehicle_width, 
        max_vehicle_width,
        min_init_dist
    )

    # beh generator
    passive = get_passive_behavior_params(
                lon_σ = lon_accel_std_dev, 
                lat_σ = lat_accel_std_dev, 
                overall_response_time = overall_response_time,
                err_p_a_to_i = err_p_a_to_i,
                err_p_i_to_a = err_p_i_to_a)
    aggressive = get_aggressive_behavior_params(
                lon_σ = lon_accel_std_dev, 
                lat_σ = lat_accel_std_dev, 
                overall_response_time = overall_response_time,
                err_p_a_to_i = err_p_a_to_i,
                err_p_i_to_a = err_p_i_to_a)

    params = [aggressive, passive]
    weights = StatsBase.Weights([.5,.5])
    behavior_gen = PredefinedBehaviorGenerator(params, weights)
    gen = FactoredGenerator(roadway_gen, scene_gen, behavior_gen)


    
    subexts = [
        CoreFeatureExtractor(),
        TemporalFeatureExtractor(),
        CarLidarFeatureExtractor(extract_carlidar_rangerate = true),
        BehavioralFeatureExtractor()
    ]
    ext = MultiFeatureExtractor(subexts)


    feature_dim = length(ext)
    target_ext = TargetExtractor()
    target_dim = length(target_ext)
    target_timesteps = Int(ceil(sampling_time / sampling_period))

    prime_time = (feature_timesteps * feature_step_size) * .1 + burn_in_time
    max_num_scenes = Int(ceil((prime_time + sampling_time) / sampling_period))
    rec = SceneRecord(max_num_scenes, sampling_period, max_num_veh)
    features = Array{Float64}(feature_dim, feature_timesteps, max_num_veh)
    targets = Array{Float64}(target_dim, target_timesteps, max_num_veh)
    agg_targets = Array{Float64}(target_dim, target_timesteps, max_num_veh)

    eval = MonteCarloEvaluator(
        ext, 
        target_ext, 
        num_runs, 
        prime_time, 
        sampling_time,
        veh_idx_can_change, 
        rec,
        features, 
        targets, 
        agg_targets, 
        feature_step_size = feature_step_size
    )

    max_num_samples = num_scenarios * max_num_veh
    feature_dim = length(ext)
    use_weights = typeof(weights) == Void ? false : true
    attrs = Dict()
    attrs["feature_names"] = feature_names(ext)
    attrs["target_names"] = feature_names(target_ext)
    dataset = Dataset(
        output_filepath, 
        feature_dim, 
        feature_timesteps, 
        target_dim, 
        target_timesteps,
        max_num_samples, 
        chunk_dim = chunk_dim, 
        init_file = false, 
        attrs = attrs
    )

    seeds = Vector{Int}() # seeds are replaced by parallel collector
    scene = Scene(max_num_veh)
    models = Dict{Int, DriverModel}()
    col = DatasetCollector(seeds, gen, eval, dataset, scene, models, roadway, id = col_id)
    return col
end

function get_filepaths(filepath, n)
    dir = dirname(filepath)
    filename = basename(filepath)
    return [string(dir, "/proc_$(i)_$(filename)") for i in 1:n]
end

function build_parallel_dataset_collector(
        num_scenarios,
        num_col,
        output_filepath
    )
    filepaths = get_filepaths(output_filepath, num_col)
    num_scenarios_each = Int(ceil(num_scenarios / num_col))
    cols = [build_dataset_collector(filepaths[i], i, num_scenarios_each) for i in 1:num_col]
    seeds = collect(1:(num_scenarios))
    pcol = ParallelDatasetCollector(cols, seeds, output_filepath)
    return pcol
end
