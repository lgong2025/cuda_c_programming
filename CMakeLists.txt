cmake_minimum_required(VERSION 3.18)



# --- Add this line to set the host compiler for CUDA ---
set(CMAKE_CUDA_HOST_COMPILER "/home/saber/nvcc_host_compiler")
project(cuda_programming LANGUAGES CXX CUDA)



# --- Automatic Target Creation ---

file(GLOB CUDASAMPLES "samples/*.cu")

foreach(sample ${CUDASAMPLES})
    get_filename_component(sample_name ${sample} NAME_WE)
    add_executable(${sample_name} ${sample})

    find_package(CUDAToolkit REQUIRED)

    set_target_properties(
        ${sample_name} PROPERTIES 
        RUNTIME_OUTPUT_DIRECTORY
        "${CMAKE_BINARY_DIR}/bin"
    )

    message(
        STATUS "Created CUDA sample: ${sample_name}"
    )

    # --- Define a Custom 'run' Target for Each Sample ---
    set(run_target "run_${sample_name}")
    add_custom_target(${run_target}
        # command 1: build the sample
        COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target ${sample_name}
        # command 2: run the sample
        COMMAND $<TARGET_FILE:${sample_name}>

        DEPENDS ${sample_name}

        # VERBATIM ensures correct quoting and argument passing
        VERBATIM 
    )

    message(STATUS "Created CUDA run target: ${run_target}")
endforeach()