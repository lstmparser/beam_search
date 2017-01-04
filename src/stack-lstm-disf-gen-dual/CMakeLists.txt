INCLUDE_DIRECTORIES(${source_directory})

ADD_EXECUTABLE(disf_stack_lstm_gen_dual lstm-parse.cc ../training_utils.cc ${SHARED_SRC})
target_link_libraries(disf_stack_lstm_gen_dual cnn ${Boost_LIBRARIES})
