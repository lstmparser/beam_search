INCLUDE_DIRECTORIES(${source_directory})

ADD_EXECUTABLE(disf_stack_new-action lstm-parse.cc ../training_utils.cc ${SHARED_SRC})
target_link_libraries(disf_stack_new-action cnn ${Boost_LIBRARIES})
