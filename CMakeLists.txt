cmake_minimum_required( VERSION 2.8 )
project( movo )
find_package(OpenCV REQUIRED)
find_package(Eigen3  REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io visualization)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})
add_executable( movo src/movo.cpp 
		     src/filters.cpp 
		     src/drawers.cpp 
		     src/conversions.cpp
		     src/paramreader.cpp
		     src/featureDetectionAndTracking.cpp 
		     src/init.cpp )
target_link_libraries(movo ${OpenCV_LIBS} ${PCL_COMMON_LIBRARIES} ${PCL_IO_LIBRARIES} 
					   ${PCL_VISUALIZATION_LIBRARIES}
					   )
