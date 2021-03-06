SET(HEADERS "")
SET(SOURCES "")

FOREACH(dir ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/details)
  APPEND_GLOB(HEADERS ${dir}/*.hpp)
  APPEND_GLOB(SOURCES ${dir}/*.cpp)
  TRIBITS_INCLUDE_DIRECTORIES(${dir})
ENDFOREACH()

TRIBITS_ADD_LIBRARY(
  dtk_search
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  DEPLIBS dtk_utils
  ADDED_LIB_TARGET_NAME_OUT DTK_SEARCH_LIBNAME
  )

# We need to set the linker language explicitly here for CUDA builds.
SET_PROPERTY(
  TARGET ${DTK_SEARCH_LIBNAME}
  APPEND PROPERTY LINKER_LANGUAGE CXX
  )
