cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(SimpleDNN
  VERSION 2.0
  DESCRIPTION "SimpleDNN v2.0 Windows/Linux"
  LANGUAGES C CXX)

set(DEFAULT_BUILD_TYPE "Release")
set(CMAKE_BUILD_TYPE "Debug")

#STNDARD STUFF
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAGS_COMMON}")
set(TARGET_NAME "SimpleDNN")

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
configure_file(config.json ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/config.json)


if (WIN32)
	configure_file(config.json ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config.json)
	configure_file(config.json ${CMAKE_BINARY_DIR}/config.json)
	configure_file(config.json ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/config.json)
endif (WIN32)
if (UNIX)
	configure_file(config.json ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/config.json)
	configure_file(config.json ${CMAKE_BINARY_DIR}/config.json)
	configure_file(config.json ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${CMAKE_BUILD_TYPE}/config.json)
endif (UNIX)

if (WIN32)
	set (CMAKE_PREFIX_PATH "E:\\repo\\opencvBuild")
	list(APPEND CMAKE_PREFIX_PATH "E:\\Qt\\5.12.8\\msvc2017_64")
	#list(APPEND CMAKE_PREFIX_PATH "E:\\repo\\ConfigReader\\build")
	#list(APPEND CMAKE_PREFIX_PATH "E:\\repo\\ProcessingModules\\build")
	list(APPEND CMAKE_PREFIX_PATH "E:\\repo\\dlib\\build\\installDebug\\lib\\cmake\\dlib")
	#find_package(ConfigReader REQUIRED)
	#find_package(ProcessingModules REQUIRED)
	#include_directories(${ProcessingModules_DIR})
endif (WIN32)

find_package(Qt5Core REQUIRED)
find_package(Qt5Widgets REQUIRED)
#find_package(OpenGL)
find_package(Threads REQUIRED)
find_package(OpenCV REQUIRED)
find_package(dlib REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
add_subdirectory(spdlog)

file(GLOB_RECURSE SRC_GLOB src/*.cc src/*.c src/*.cpp)
file(GLOB_RECURSE HEADERS_GLOB src/*.h src/*.hpp)
  

set(SRC
	ConfigReader/src/configreader.cc
)

set(HEADERS
	ConfigReader/include/configreader.h
)
add_executable(${TARGET_NAME}
    ${SRC} ${HEADERS}
    ${SRC_GLOB} ${HEADERS_GLOB}
)
qt5_use_modules(${PROJECT_NAME} Widgets)

if (UNIX)
	target_link_libraries(${PROJECT_NAME} spdlog::spdlog)
	target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${QT_LIBRARIES} Threads::Threads)
	target_link_libraries(${PROJECT_NAME} dlib::dlib)
endif (UNIX)

if (WIN32)
	target_link_libraries(${PROJECT_NAME} spdlog::spdlog)
	target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${QT_LIBRARIES} Threads::Threads)
	#target_link_libraries(${PROJECT_NAME} ConfigReader)
	target_link_libraries(${PROJECT_NAME} dlib::dlib)
endif (WIN32)

message("CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE} ")
if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND WIN32)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

endif()
if (CMAKE_BUILD_TYPE STREQUAL "Release" AND WIN32)

set (CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

endif()


if (WIN32)
    get_target_property(QT5_QMAKE_EXECUTABLE Qt5::qmake IMPORTED_LOCATION)
    get_filename_component(QT5_WINDEPLOYQT_EXECUTABLE ${QT5_QMAKE_EXECUTABLE} PATH)
    set(QT5_WINDEPLOYQT_EXECUTABLE "${QT5_WINDEPLOYQT_EXECUTABLE}/windeployqt.exe")

    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
       COMMAND ${QT5_WINDEPLOYQT_EXECUTABLE} --qmldir ${CMAKE_SOURCE_DIR} $<TARGET_FILE_DIR:${PROJECT_NAME}>)

endif(WIN32)

if (WIN32)


	message("CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE} ")

	message("Copy OpenCV file:" )
	if(${CMAKE_BUILD_TYPE} STREQUAL  "Release")
		set(OPENCV_PREFIX_BUILD_TYPE "")
	endif()
	if(${CMAKE_BUILD_TYPE} STREQUAL  "Debug")
		set(OPENCV_PREFIX_BUILD_TYPE "d")
	endif()


	add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${OpenCV_DIR}/bin/${CMAKE_BUILD_TYPE}/opencv_stitching450${OPENCV_PREFIX_BUILD_TYPE}.dll   ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/
	COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${OpenCV_DIR}/bin/${CMAKE_BUILD_TYPE}/opencv_bgsegm450${OPENCV_PREFIX_BUILD_TYPE}.dll   ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/
	COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${OpenCV_DIR}/bin/${CMAKE_BUILD_TYPE}/opencv_tracking450${OPENCV_PREFIX_BUILD_TYPE}.dll   ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/
	COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${OpenCV_DIR}/bin/${CMAKE_BUILD_TYPE}/opencv_videoio450${OPENCV_PREFIX_BUILD_TYPE}.dll   ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/
	COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${OpenCV_DIR}/bin/${CMAKE_BUILD_TYPE}/opencv_video450${OPENCV_PREFIX_BUILD_TYPE}.dll   ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/
	COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${OpenCV_DIR}/bin/${CMAKE_BUILD_TYPE}/opencv_imgcodecs450${OPENCV_PREFIX_BUILD_TYPE}.dll   ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/
	COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${OpenCV_DIR}/bin/${CMAKE_BUILD_TYPE}/opencv_imgproc450${OPENCV_PREFIX_BUILD_TYPE}.dll   ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/
	COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${OpenCV_DIR}/bin/${CMAKE_BUILD_TYPE}/opencv_core450${OPENCV_PREFIX_BUILD_TYPE}.dll   ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/
	COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${OpenCV_DIR}/bin/${CMAKE_BUILD_TYPE}/opencv_calib3d450${OPENCV_PREFIX_BUILD_TYPE}.dll   ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/
	COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${OpenCV_DIR}/bin/${CMAKE_BUILD_TYPE}/opencv_features2d450${OPENCV_PREFIX_BUILD_TYPE}.dll   ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/
	COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${OpenCV_DIR}/bin/${CMAKE_BUILD_TYPE}/opencv_flann450${OPENCV_PREFIX_BUILD_TYPE}.dll   ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/
	COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${OpenCV_DIR}/bin/${CMAKE_BUILD_TYPE}/opencv_dnn450${OPENCV_PREFIX_BUILD_TYPE}.dll   ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/
	COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${OpenCV_DIR}/bin/${CMAKE_BUILD_TYPE}/opencv_highgui450${OPENCV_PREFIX_BUILD_TYPE}.dll   ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/
	COMMAND ${CMAKE_COMMAND} -E copy_if_different  ${OpenCV_DIR}/bin/${CMAKE_BUILD_TYPE}/opencv_highgui450${OPENCV_PREFIX_BUILD_TYPE}.dll   ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/
	)
endif(WIN32)