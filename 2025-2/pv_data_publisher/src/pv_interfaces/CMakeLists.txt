cmake_minimum_required(VERSION 3.8)
project(pv_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PVSample.msg"
  DEPENDENCIES builtin_interfaces
  # "msg/PVFeature.msg"
  # "msg/PVAnomaly.msg"
)
  
ament_export_dependencies(rosidl_default_runtime)

ament_package()
