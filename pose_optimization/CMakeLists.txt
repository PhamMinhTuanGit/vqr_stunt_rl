cmake_minimum_required(VERSION 3.19)

project(vqr_pose_optimization LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Pinocchio installed from the Python "pin" package places its CMake package in
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
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

get_filename_component(VQR_REPOSITORY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." REALPATH)
set(VQR_URDF_PATH
  "${VQR_REPOSITORY_ROOT}/deep_robotics_model/VQRWheel/VQRWheel_urdf/urdf/VQRWheel.urdf")
set(VQR_USD_CONVERTER_CONFIG_PATH
  "${VQR_REPOSITORY_ROOT}/deep_robotics_model/VQRWheel/VQRWheel_usd/config.yaml")
set(VQR_ROBOT_CONFIG_PATH
  "${VQR_REPOSITORY_ROOT}/source/rl_training/rl_training/tasks/manager_based/locomotion/pivot/config/wheeled/vqr/robot_cfg.py")
set(VQR_CASADI_PYTHON_PATH
  "${VQR_REPOSITORY_ROOT}/third_party/casadi")
set(VQR_EQUILIBRIUM_CONFIG_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/config/equilibrium.yaml")
set(VQR_KINEMATIC_OUTPUT_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/output/kinematic_equilibrium_FL_HR.yaml")
set(VQR_STATIC_OUTPUT_PATH
  "${CMAKE_CURRENT_SOURCE_DIR}/output/static_equilibrium_FL_HR.yaml")

foreach(_required_file IN ITEMS
    "${VQR_URDF_PATH}"
    "${VQR_USD_CONVERTER_CONFIG_PATH}"
    "${VQR_ROBOT_CONFIG_PATH}")
  if(NOT EXISTS "${_required_file}")
    message(FATAL_ERROR "Required VQR model provenance file does not exist: ${_required_file}")
  endif()
endforeach()

if(NOT EXISTS "${VQR_CASADI_PYTHON_PATH}/casadi/__init__.py")
  message(FATAL_ERROR
    "CasADi is missing from ${VQR_CASADI_PYTHON_PATH}. See README.md for the "
    "repository-local installation command.")
endif()

get_filename_component(_pinocchio_cmake_parent "${pinocchio_DIR}" DIRECTORY)
get_filename_component(_pinocchio_lib_parent "${_pinocchio_cmake_parent}" DIRECTORY)
get_filename_component(VQR_CMEEL_PREFIX "${_pinocchio_lib_parent}" DIRECTORY)
file(GLOB _pinocchio_python_candidates LIST_DIRECTORIES TRUE
  "${VQR_CMEEL_PREFIX}/lib/python*/site-packages")
foreach(_python_path IN LISTS _pinocchio_python_candidates)
  file(GLOB _pinocchio_bindings
    "${_python_path}/pinocchio/pinocchio_pywrap*.so")
  if(_pinocchio_bindings)
    set(VQR_PINOCCHIO_PYTHON_PATH "${_python_path}")
    break()
  endif()
endforeach()
if(NOT VQR_PINOCCHIO_PYTHON_PATH)
  message(FATAL_ERROR
    "The Python bindings matching ${pinocchio_DIR} were not found.")
endif()
string(REGEX REPLACE
  "/lib/python[^/]+/site-packages/cmeel.prefix$" ""
  VQR_PYTHON_ENV_ROOT "${VQR_CMEEL_PREFIX}")
set(VQR_PYTHON_EXECUTABLE "${VQR_PYTHON_ENV_ROOT}/bin/python")
if(NOT EXISTS "${VQR_PYTHON_EXECUTABLE}")
  message(FATAL_ERROR
    "Python executable for Pinocchio was not found: ${VQR_PYTHON_EXECUTABLE}")
endif()

add_library(wheel_contact_geometry src/wheel_contact_geometry.cpp)
target_include_directories(wheel_contact_geometry PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(wheel_contact_geometry PUBLIC
  Eigen3::Eigen urdfdom::urdfdom_model)
target_compile_options(wheel_contact_geometry PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(wheel_contact_geometry PROPERTIES
  POSITION_INDEPENDENT_CODE ON)

add_library(tread_contact_c_api SHARED src/wheel_contact_geometry_c_api.cpp)
target_link_libraries(tread_contact_c_api PRIVATE wheel_contact_geometry)
target_compile_options(tread_contact_c_api PRIVATE
  -Wall -Wextra -Wpedantic)

add_library(kinematic_geometry src/kinematic_geometry.cpp)
target_include_directories(kinematic_geometry PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(kinematic_geometry PUBLIC Eigen3::Eigen)
target_compile_options(kinematic_geometry PRIVATE -Wall -Wextra -Wpedantic)

add_executable(model_audit src/model_audit.cpp)
target_link_libraries(model_audit PRIVATE
  pinocchio::pinocchio wheel_contact_geometry)
target_compile_definitions(model_audit PRIVATE
  VQR_URDF_PATH="${VQR_URDF_PATH}"
  VQR_USD_CONVERTER_CONFIG_PATH="${VQR_USD_CONVERTER_CONFIG_PATH}"
  VQR_ROBOT_CONFIG_PATH="${VQR_ROBOT_CONFIG_PATH}"
)
target_compile_options(model_audit PRIVATE -Wall -Wextra -Wpedantic)

add_executable(validate_kinematic_equilibrium
  src/validate_kinematic_equilibrium.cpp)
target_link_libraries(validate_kinematic_equilibrium PRIVATE
  pinocchio::pinocchio
  wheel_contact_geometry
  yaml-cpp)
target_compile_definitions(validate_kinematic_equilibrium PRIVATE
  VQR_URDF_PATH="${VQR_URDF_PATH}"
  VQR_KINEMATIC_OUTPUT_PATH="${VQR_KINEMATIC_OUTPUT_PATH}")
target_compile_options(validate_kinematic_equilibrium PRIVATE
  -Wall -Wextra -Wpedantic)

add_executable(validate_static_equilibrium
  src/validate_static_equilibrium.cpp)
target_link_libraries(validate_static_equilibrium PRIVATE
  pinocchio::pinocchio
  wheel_contact_geometry
  yaml-cpp)
target_compile_definitions(validate_static_equilibrium PRIVATE
  VQR_URDF_PATH="${VQR_URDF_PATH}"
  VQR_STATIC_OUTPUT_PATH="${VQR_STATIC_OUTPUT_PATH}")
target_compile_options(validate_static_equilibrium PRIVATE
  -Wall -Wextra -Wpedantic)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/apps/kinematic_equilibrium.in"
  "${CMAKE_CURRENT_BINARY_DIR}/kinematic_equilibrium"
  @ONLY)
file(CHMOD "${CMAKE_CURRENT_BINARY_DIR}/kinematic_equilibrium"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/apps/static_equilibrium.in"
  "${CMAKE_CURRENT_BINARY_DIR}/static_equilibrium"
  @ONLY)
file(CHMOD "${CMAKE_CURRENT_BINARY_DIR}/static_equilibrium"
  PERMISSIONS
    OWNER_READ OWNER_WRITE OWNER_EXECUTE
    GROUP_READ GROUP_EXECUTE
    WORLD_READ WORLD_EXECUTE)

include(CTest)
if(BUILD_TESTING)
  add_test(NAME vqr_model_audit COMMAND model_audit)

  add_executable(wheel_contact_geometry_test
    tests/wheel_contact_geometry_test.cpp)
  target_link_libraries(wheel_contact_geometry_test PRIVATE
    wheel_contact_geometry)
  target_compile_definitions(wheel_contact_geometry_test PRIVATE
    VQR_URDF_PATH="${VQR_URDF_PATH}")
  target_compile_options(wheel_contact_geometry_test PRIVATE
    -Wall -Wextra -Wpedantic)
  add_test(NAME wheel_contact_geometry COMMAND wheel_contact_geometry_test)

  add_executable(kinematic_geometry_test
    tests/kinematic_geometry_test.cpp)
  target_link_libraries(kinematic_geometry_test PRIVATE kinematic_geometry)
  target_compile_options(kinematic_geometry_test PRIVATE
    -Wall -Wextra -Wpedantic)
  add_test(NAME kinematic_geometry COMMAND kinematic_geometry_test)

  add_test(NAME solve_kinematic_equilibrium
    COMMAND "${CMAKE_CURRENT_BINARY_DIR}/kinematic_equilibrium")
  add_test(NAME verify_kinematic_equilibrium
    COMMAND "${CMAKE_CURRENT_BINARY_DIR}/kinematic_equilibrium"
      --verify --output "${VQR_KINEMATIC_OUTPUT_PATH}")
  set_tests_properties(verify_kinematic_equilibrium PROPERTIES
    DEPENDS solve_kinematic_equilibrium)

  add_test(NAME independent_kinematic_validation
    COMMAND validate_kinematic_equilibrium
      "${VQR_KINEMATIC_OUTPUT_PATH}")

  add_test(NAME solve_static_equilibrium
    COMMAND "${CMAKE_CURRENT_BINARY_DIR}/static_equilibrium")
  add_test(NAME independent_static_validation
    COMMAND validate_static_equilibrium
      "${VQR_STATIC_OUTPUT_PATH}")
  set_tests_properties(independent_static_validation PROPERTIES
    DEPENDS solve_static_equilibrium)
endif()
