cmake_minimum_required(VERSION 3.16)

project(vqr_pose_optimization_model_audit LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Pinocchio installed from the Python "pin" wheel places its CMake package in
# cmeel.prefix.  Try the normal search first, then discover such prefixes so
# the requested plain `cmake -S . -B build` command works in this repository.
find_package(pinocchio 2.7 CONFIG QUIET)
if(NOT pinocchio_FOUND)
  file(GLOB _cmeel_prefix_candidates LIST_DIRECTORIES TRUE
    "$ENV{CONDA_PREFIX}/lib/python*/site-packages/cmeel.prefix"
    "$ENV{HOME}/miniconda3/envs/*/lib/python*/site-packages/cmeel.prefix"
    "$ENV{HOME}/.local/lib/python*/site-packages/cmeel.prefix"
  )
  foreach(_prefix IN LISTS _cmeel_prefix_candidates)
    if(EXISTS "${_prefix}/lib/cmake/pinocchio/pinocchioConfig.cmake")
      list(PREPEND CMAKE_PREFIX_PATH "${_prefix}")
      find_package(pinocchio 2.7 CONFIG QUIET)
      if(pinocchio_FOUND)
        message(STATUS "Found Pinocchio through cmeel prefix: ${_prefix}")
        break()
      endif()
    endif()
  endforeach()
endif()
if(NOT pinocchio_FOUND)
  message(FATAL_ERROR
    "Pinocchio >= 2.7 was not found. Activate the project environment or set "
    "CMAKE_PREFIX_PATH to the installation prefix containing pinocchioConfig.cmake.")
endif()

find_package(urdfdom REQUIRED)

get_filename_component(VQR_REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." REALPATH)
set(VQR_URDF_PATH
  "${VQR_REPOSITORY_ROOT}/deep_robotics_model/VQRWheel/VQRWheel_urdf/urdf/VQRWheel.urdf")
set(VQR_USD_CONVERTER_CONFIG_PATH
  "${VQR_REPOSITORY_ROOT}/deep_robotics_model/VQRWheel/VQRWheel_usd/config.yaml")
set(VQR_ROBOT_CONFIG_PATH
  "${VQR_REPOSITORY_ROOT}/source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/robot_cfg.py")

foreach(_required_file IN ITEMS
    "${VQR_URDF_PATH}"
    "${VQR_USD_CONVERTER_CONFIG_PATH}"
    "${VQR_ROBOT_CONFIG_PATH}")
  if(NOT EXISTS "${_required_file}")
    message(FATAL_ERROR "Required VQR model provenance file does not exist: ${_required_file}")
  endif()
endforeach()

add_executable(model_audit src/model_audit.cpp)
target_link_libraries(model_audit PRIVATE pinocchio::pinocchio urdfdom::urdfdom_model)
target_compile_definitions(model_audit PRIVATE
  VQR_URDF_PATH="${VQR_URDF_PATH}"
  VQR_USD_CONVERTER_CONFIG_PATH="${VQR_USD_CONVERTER_CONFIG_PATH}"
  VQR_ROBOT_CONFIG_PATH="${VQR_ROBOT_CONFIG_PATH}"
)
target_compile_options(model_audit PRIVATE -Wall -Wextra -Wpedantic)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME vqr_model_audit COMMAND model_audit)
endif()

