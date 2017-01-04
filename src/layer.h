#ifndef __LAYER_H__
#define __LAYER_H__

#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <set>
#include "cnn/cnn.h"
#include "cnn/rnn.h"
#include "cnn/lstm.h"
#include "cnn/expr.h"
#include "cnn/nodes.h"

struct SymbolEmbedding {
  cnn::LookupParameters* p_labels;

  // Use to store the embedding for label.
  SymbolEmbedding(cnn::Model* m, unsigned n, unsigned dim);
  cnn::expr::Expression embed(cnn::ComputationGraph* cg, unsigned label_id);
};

struct ConstSymbolEmbedding {
  cnn::LookupParameters* p_labels;

  // Use to store the embedding for label.
  ConstSymbolEmbedding(cnn::Model* m, unsigned n, unsigned dim);
  cnn::expr::Expression embed(cnn::ComputationGraph* cg, unsigned label_id);
};


struct BinnedDistanceEmbedding {
  cnn::LookupParameters* p_e;
  unsigned max_bin;

  BinnedDistanceEmbedding(cnn::Model* m, unsigned hidden, unsigned n_bin = 8);
  cnn::expr::Expression embed(cnn::ComputationGraph* cg, int distance);
};


struct BinnedDurationEmbedding {
  cnn::LookupParameters* p_e;
  unsigned max_bin;

  BinnedDurationEmbedding(cnn::Model* m, unsigned hidden, unsigned n_bin = 8);
  cnn::expr::Expression embed(cnn::ComputationGraph* cg, unsigned dur);
};


struct StaticInputLayer {
  cnn::LookupParameters* p_w;  // Word embedding
  cnn::LookupParameters* p_p;  // Postag embedding
  cnn::LookupParameters* p_t;  // Pretrained word embedding

  cnn::Parameters* p_ib;
  cnn::Parameters* p_w2l;
  cnn::Parameters* p_p2l;
  cnn::Parameters* p_t2l;
  cnn::Parameters* p_g2l; // parameter for gen feature

  bool use_word;
  bool use_postag;
  bool use_pretrained_word;

  StaticInputLayer(cnn::Model* model,unsigned dim_gen,
    unsigned size_word, unsigned dim_word,
    unsigned size_postag, unsigned dim_postag,
    unsigned size_pretrained_word, unsigned dim_pretrained_word,
    unsigned dim_output,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  cnn::expr::Expression add_input(cnn::ComputationGraph* hg,
    unsigned wid, unsigned pid, unsigned pre_wid,const std::vector<cnn::real> current_gen_feature);
};


struct DynamicInputLayer : public StaticInputLayer {
  cnn::LookupParameters* p_l;
  cnn::Parameters* p_l2l;
  bool use_label;

  DynamicInputLayer(cnn::Model* model,unsigned dim_gen,
    unsigned size_word, unsigned dim_word,
    unsigned size_postag, unsigned dim_postag,
    unsigned size_pretrained_word, unsigned dim_pretrained_word,
    unsigned size_label, unsigned dim_label,
    unsigned dim_output,
    const std::unordered_map<unsigned, std::vector<float>>& pretrained);

  cnn::expr::Expression add_input2(cnn::ComputationGraph* hg, 
    unsigned wid, unsigned pid, unsigned pre_wid, unsigned lid);

  cnn::expr::Expression add_input2(cnn::ComputationGraph* hg, 
    unsigned wid, unsigned pid, unsigned pre_wid, cnn::expr::Expression& expr);
};


struct LSTMLayer {
  unsigned n_items;
  cnn::LSTMBuilder lstm;
  cnn::Parameters* p_guard;
  bool reversed;

  LSTMLayer(cnn::Model* model, unsigned n_layers, unsigned dim_input, unsigned dim_hidden, bool reversed = false);
  void new_graph(cnn::ComputationGraph* hg);
  void add_inputs(cnn::ComputationGraph* hg, const std::vector<cnn::expr::Expression>& exprs,const std::vector<cnn::expr::Expression>& encdec);
  cnn::expr::Expression get_output(cnn::ComputationGraph* hg, int index);
  void get_outputs(cnn::ComputationGraph* hg, std::vector<cnn::expr::Expression>& outputs);
  void set_dropout(float& rate);
  void disable_dropout();
};



struct RNNLanguageModel {
    cnn::LookupParameters* p_c;
    cnn::Parameters* p_R;
    cnn::Parameters* p_bias;
    cnn::LSTMBuilder builders;
    RNNLanguageModel(cnn::Model* model_lm,unsigned VOCAB_SIZE,unsigned LAYERS, unsigned INPUT_DIM, unsigned HIDDEN_DIM);

    void BuildLMGraphs(const std::vector<std::vector<unsigned>>& sents,
                             unsigned id,
                             unsigned & chars,
                             unsigned bsize,
                             cnn::ComputationGraph* hg) ;
    void set_dropout(float& rate);
    void disable_dropout();
    cnn::expr::Expression new_graph(cnn::ComputationGraph* hg, unsigned sos);
    cnn::expr::Expression add_input(cnn::ComputationGraph* hg, unsigned word);
    cnn::expr::Expression get_iR(cnn::ComputationGraph* hg);

//    cnn::expr::Expression add_input(cnn::ComputationGraph* hg,
//                                    unsigned wid, unsigned pid, unsigned pre_wid,const std::vector<cnn::real> current_gen_feature);
//



//    {
//      const unsigned slen = sents[id].size();
//      if (apply_dropout) {
//        builder.set_dropout(DROPOUT);
//      } else {
//        builder.disable_dropout();
//      }
//      builder.new_graph(cg);  // reset RNN builder for new graph
//      builder.start_new_sequence();
//
//      Expression i_R = parameter(cg, p_R); // hidden -> word rep parameter
//      Expression i_bias = parameter(cg, p_bias);  // word bias
//      vector<Expression> errs;
//      vector<unsigned> last_arr(bsize, sents[0][0]), next_arr(bsize);
//      for (unsigned t = 1; t < slen; ++t) {
//        for (unsigned i = 0; i < bsize; ++i) {
//          next_arr[i] = sents[id+i][t];
//          if(next_arr[i] != *sents[id].rbegin()) chars++; // add non-EOS
//        }
//        // y_t = RNN(x_t)
//        Expression i_x_t = lookup(cg, p_c, last_arr);
//        Expression i_y_t = builder.add_input(i_x_t);
//        Expression i_r_t = i_bias + i_R * i_y_t;
//        Expression i_err = pickneglogsoftmax(i_r_t, next_arr);
//        errs.push_back(i_err);
//        last_arr = next_arr;
//      }
//      Expression i_nerr = sum_batches(sum(errs));
//      return i_nerr;
//    }

//    // return Expression for total loss
//    void RandomSample(int max_len = 150) {
//      cerr << endl;
//      ComputationGraph cg;
//      builder.new_graph(cg);  // reset RNN builder for new graph
//      builder.start_new_sequence();
//
//      Expression i_R = parameter(cg, p_R);
//      Expression i_bias = parameter(cg, p_bias);
//      vector<Expression> errs;
//      int len = 0;
//      int cur = kSOS;
//      while(len < max_len && cur != kEOS) {
//        ++len;
//        Expression i_x_t = lookup(cg, p_c, cur);
//        // y_t = RNN(x_t)
//        Expression i_y_t = builder.add_input(i_x_t);
//        Expression i_r_t = i_bias + i_R * i_y_t;
//
//        Expression ydist = softmax(i_r_t);
//
//        unsigned w = 0;
//        while (w == 0 || (int)w == kSOS) {
//          auto dist = as_vector(cg.incremental_forward());
//          double p = rand01();
//          for (; w < dist.size(); ++w) {
//            p -= dist[w];
//            if (p < 0.0) { break; }
//          }
//          if (w == dist.size()) w = kEOS;
//        }
//        cerr << (len == 1 ? "" : " ") << d.Convert(w);
//        cur = w;
//      }
//      cerr << endl;
//    }
};







struct BidirectionalLSTMLayer {
  typedef std::pair<cnn::expr::Expression, cnn::expr::Expression> Output;
  unsigned n_items;
  cnn::LSTMBuilder fw_lstm;
  cnn::LSTMBuilder bw_lstm;
  cnn::Parameters* p_fw_guard;
  cnn::Parameters* p_bw_guard;

  BidirectionalLSTMLayer(cnn::Model* model,
    unsigned n_lstm_layers,
    unsigned dim_lstm_input,
    unsigned dim_hidden);

  void new_graph(cnn::ComputationGraph* hg);
  void add_inputs(cnn::ComputationGraph* hg, const std::vector<cnn::expr::Expression>& exprs);
  Output get_output(cnn::ComputationGraph* hg, int index);
  void get_outputs(cnn::ComputationGraph* hg, std::vector<Output>& outputs);
  void set_dropout(float& rate);
  void disable_dropout();
};


struct SoftmaxLayer {
  cnn::Parameters* p_B;
  cnn::Parameters* p_W;

  SoftmaxLayer(cnn::Model* model, unsigned dim_input, unsigned dim_output);
  cnn::expr::Expression get_output(cnn::ComputationGraph* hg,
    const cnn::expr::Expression& expr);
};


struct DenseLayer {
  cnn::Parameters *p_W, *p_B;
  DenseLayer(cnn::Model* model, unsigned dim_input, unsigned dim_output);
  cnn::expr::Expression get_output(cnn::ComputationGraph* hg,
    const cnn::expr::Expression& expr);
};








struct Merge2Layer {
  cnn::Parameters *p_B, *p_W1, *p_W2;

  Merge2Layer(cnn::Model* model,
    unsigned dim_input1,
    unsigned dim_input2,
    unsigned dim_output);

  cnn::expr::Expression get_output(cnn::ComputationGraph* hg,
    const cnn::expr::Expression& expr1,
    const cnn::expr::Expression& expr2);
};

struct Merge_enc_Layer {
    cnn::Parameters *p_ie2h;
    cnn::Parameters *p_bie;
    cnn::Parameters *p_h2oe;
    cnn::Parameters *p_boe;
    unsigned layers;
    unsigned hidden_dim;

  Merge_enc_Layer(cnn::Model* model,
    unsigned HIDDEN_DIM,
    unsigned LAYERS);

  std::vector<cnn::expr::Expression> get_output(cnn::ComputationGraph* hg,
                                   const cnn::expr::Expression& expr1);
};


struct Merge3Layer {
  cnn::Parameters *p_B, *p_W1, *p_W2;
  //std::vector<cnn::Parameters*>  p_W3;
  cnn::LookupParameters* p_W3;

  Merge3Layer(cnn::Model* model,
    unsigned word_dim,
    unsigned dim_input1,
    unsigned dim_input2,
    unsigned dim_input3,
    unsigned dim_output);

  cnn::expr::Expression get_output(cnn::ComputationGraph* hg,
    unsigned wid,
    const cnn::expr::Expression& expr1,
    const cnn::expr::Expression& expr2,
    const cnn::expr::Expression& expr3);
  void W3_initial(cnn::ComputationGraph *hg,  const cnn::expr::Expression& expr1);
};

//
//struct Merge4Layer {
//    cnn::Parameters *p_B, *p_W1, *p_W2;
//    cnn::Parameters *p_W3;
//    cnn::Parameters  *p_W4;
//
//    Merge4Layer(cnn::Model* model,
//                unsigned dim_input1,
//                unsigned dim_input2,
//                unsigned dim_input3,
//                unsigned dim_input4,
//                unsigned dim_output);
//
//    cnn::expr::Expression get_output(cnn::ComputationGraph* hg,
//                                     const cnn::expr::Expression& expr1,
//                                     const cnn::expr::Expression& expr2,
//                                     const cnn::expr::Expression& expr3,
//                                     const cnn::expr::Expression& expr4);
//};


struct Merge4Layer {
  cnn::Parameters *p_B, *p_W1, *p_W2;
  //std::vector<cnn::Parameters*>  p_W3;
  cnn::LookupParameters* p_W3;
  cnn::Parameters  *p_W4;

  Merge4Layer(cnn::Model* model,
    unsigned word_dim,
    unsigned dim_input1,
    unsigned dim_input2,
    unsigned dim_input3,
    unsigned dim_input4,
    unsigned dim_output);

  cnn::expr::Expression get_output(cnn::ComputationGraph* hg,
    unsigned wid,
    const cnn::expr::Expression& expr1,
    const cnn::expr::Expression& expr2,
    const cnn::expr::Expression& expr3,
    const cnn::expr::Expression& expr4);
  void W3_initial(cnn::ComputationGraph *hg, const cnn::expr::Expression& expr1);
};

struct Merge5Layer {
  cnn::Parameters *p_B, *p_W1, *p_W2, *p_W3, *p_W4, *p_W5;

  Merge5Layer(cnn::Model* model,
    unsigned dim_input1,
    unsigned dim_input2,
    unsigned dim_input3,
    unsigned dim_input4,
    unsigned dim_input5,
    unsigned dim_output);

  cnn::expr::Expression get_output(cnn::ComputationGraph* hg,
    const cnn::expr::Expression& expr1,
    const cnn::expr::Expression& expr2,
    const cnn::expr::Expression& expr3,
    const cnn::expr::Expression& expr4,
    const cnn::expr::Expression& expr5);
};


struct Merge6Layer {
  cnn::Parameters *p_B, *p_W1, *p_W2, *p_W3, *p_W4, *p_W5, *p_W6;

  Merge6Layer(cnn::Model* model,
    unsigned dim_input1,
    unsigned dim_input2,
    unsigned dim_input3,
    unsigned dim_input4,
    unsigned dim_input5,
    unsigned dim_input6,
    unsigned dim_output);

  cnn::expr::Expression get_output(cnn::ComputationGraph* hg,
    const cnn::expr::Expression& expr1,
    const cnn::expr::Expression& expr2,
    const cnn::expr::Expression& expr3,
    const cnn::expr::Expression& expr4,
    const cnn::expr::Expression& expr5,
    const cnn::expr::Expression& expr6);
};


struct SegUniEmbedding {
  // uni-directional segment embedding.
  cnn::Parameters* p_h0;
  cnn::LSTMBuilder builder;
  std::vector<std::vector<cnn::expr::Expression>> h;
  unsigned len;

  explicit SegUniEmbedding(cnn::Model& m,
    unsigned n_layers, unsigned lstm_input_dim, unsigned seg_dim);
  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, int max_seg_len = 0);
  const cnn::expr::Expression& operator()(unsigned i, unsigned j) const;
  void set_dropout(float& rate);
  void disable_dropout();
};


struct SegBiEmbedding {
  typedef std::pair<cnn::expr::Expression, cnn::expr::Expression> ExpressionPair;
  SegUniEmbedding fwd, bwd;
  std::vector<std::vector<ExpressionPair>> h;
  unsigned len;

  explicit SegBiEmbedding(cnn::Model& m,
    unsigned n_layers, unsigned lstm_input_dim, unsigned seg_dim);
  void construct_chart(cnn::ComputationGraph& cg,
    const std::vector<cnn::expr::Expression>& c, int max_seg_len = 0);
  const ExpressionPair& operator()(unsigned i, unsigned j) const;
  void set_dropout(float& rate);
  void disable_dropout();
};

#endif  //  end for __LAYER_H__
