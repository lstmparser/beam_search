INCLUDE_DIRECTORIES(${source_directory})

ADD_EXECUTABLE(disf_stack_rev lstm-parse.cc ${SHARED_SRC})
target_link_libraries(disf_stack_rev cnn ${Boost_LIBRARIES})
