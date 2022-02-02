# Their Code

- `scaled_features`: path where all words converge to the base token
  - `[0]` is base token
  - `[-1]` is original word token

## Baseline Ideas

aus dem artikel:
- furthest embedding: embedding hat in allen dims den min oder max wert, je nachdem was der wert des words in der dimension ist
- blurred embedding: nimm embedding vom satz, wende gaussian blur darauf an (embeddings der wörter werden ähnlicher)
- uniform: random embeddings
- gaussian: random embeddings um die originalen embeddings herum
  - diskretisieren macht hier wohl keinen sinn

aus dem DIG paper:
- pad token

- furthest word: wie furthest embedding, aber suche von dort das nächste echte wort
- zero embedding
- average word embedding: durchschnitt aller embeddings aus dem vocabulary (möglicherweise gewichtet nach word frequency?)
- average word: wie average word embedding, aber suche von dort aus das nächste echte wort
- random word: take for each token a random word embedding