INCLUDE_DIRECTORIES(${source_directory})

ADD_EXECUTABLE(disf_stack_beam_search lstm-parse.cc ../training_utils.cc ${SHARED_SRC})
target_link_libraries(disf_stack_beam_search cnn ${Boost_LIBRARIES})
