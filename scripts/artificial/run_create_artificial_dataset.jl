
include("create_artificial_dataset.jl")

function main()
    output_filepath = "../../data/trajectories/artificial.h5"
    num_scenarios = 10000
    num_col = 20
    pcol = build_parallel_dataset_collector(
        num_scenarios,
        num_col,
        output_filepath
    )
    generate_dataset(pcol)
    analyze_risk_dataset(output_filepath)
end

@time main()