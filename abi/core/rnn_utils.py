
import abi.core.rnn_cells as rnn_cells

def _build_recurrent_cell(hidden_dim, dropout_keep_prob):
    return rnn_cells.LayerNormLSTMCell(
        hidden_dim, 
        use_recurrent_dropout=True,
        dropout_keep_prob=dropout_keep_prob
    )