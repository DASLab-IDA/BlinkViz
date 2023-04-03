from transformers import pipeline, AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline, T5Model


#tokenized_code = "select time ( col0 ) from tab0"
tokenized_code = "select count(*) from flights where origin='JFK'"

tokenizer = AutoTokenizer.from_pretrained('SEBIS/code_trans_t5_small_source_code_summarization_sql_multitask_finetune')
model =T5Model.from_pretrained("SEBIS/code_trans_t5_small_source_code_summarization_sql_multitask_finetune")
text =tokenized_code
encoded_input = tokenizer(text, return_tensors='pt').input_ids
print(encoded_input)
decoder_input_ids = tokenizer("Studies show that", return_tensors="pt").input_ids  # Batch size 1

# forward pass
outputs = model(input_ids=encoded_input, decoder_input_ids=decoder_input_ids)
print(outputs.encoder_last_hidden_state)
print(outputs.encoder_last_hidden_state.shape)
#print(outputs)
#print(outputs.shape)
"""
# 1. Load the token classification pipeline and load it into the GPU if avilabile
pipeline = SummarizationPipeline(
    model=AutoModelWithLMHead.from_pretrained("SEBIS/code_trans_t5_small_source_code_summarization_sql_multitask_finetune"),
    tokenizer=AutoTokenizer.from_pretrained("SEBIS/code_trans_t5_small_source_code_summarization_sql_multitask_finetune", skip_special_tokens=True),
    device=0
)


# 2. Give the code for summarization, parse and tokenize it
# code = "select time (fieldname) from tablename" #@param {type:"raw"}
code = "select time from tablename" #@param {type:"raw"}
import re
import sqlparse

scanner=re.Scanner([
  (r"[^]*\]",       lambda scanner,token: token),
  (r"\+",      lambda scanner,token:"R_PLUS"),
  (r"\*",        lambda scanner,token:"R_KLEENE"),
  (r"%",        lambda scanner,token:"R_WILD"),
  (r"\^",        lambda scanner,token:"R_START"),
  (r"$",        lambda scanner,token:"R_END"),
  (r"\?",        lambda scanner,token:"R_QUESTION"),
  (r"[\.~``;_a-zA-Z0-9\s=:\{\}\-\]+", lambda scanner,token:"R_FREE"),
  (r'.', lambda scanner, token: None),
])

def tokenizeRegex(s):
  results, remainder=scanner.scan(s)
  return results

def my_traverse(token_list, statement_list, result_list):
  for t in token_list:
    if t.ttype == None:
      my_traverse(t, statement_list, result_list)
    elif t.ttype != sqlparse.tokens.Whitespace:
      statement_list.append(t.ttype)
      result_list.append(str(t))
  return statement_list, result_list

def sanitizeSql(sql):
  s = sql.strip().lower()
  if not s[-1] == ";":
    s += ';'
  s = re.sub(r'\(', r' ( ', s)
  s = re.sub(r'\)', r' ) ', s)
  s = s.replace('#', '')
  return s

statement_list = []
result_list = []
code = sanitizeSql(code)
tokens = sqlparse.parse(code)
statements, result = my_traverse(tokens, statement_list, result_list)

table_map = {}
column_map = {}
for i in range(len(statements)):
  if statements[i] in [sqlparse.tokens.Number.Integer, sqlparse.tokens.Literal.Number.Integer]:
    result[i] = "CODE_INTEGER"
  elif statements[i] in [sqlparse.tokens.Number.Float, sqlparse.tokens.Literal.Number.Float]:
    result[i] = "CODE_FLOAT"
  elif statements[i] in [sqlparse.tokens.Number.Hexadecimal, sqlparse.tokens.Literal.Number.Hexadecimal]:
    result[i] = "CODE_HEX"
  elif statements[i] in [sqlparse.tokens.String.Symbol, sqlparse.tokens.String.Single, sqlparse.tokens.Literal.String.Single, sqlparse.tokens.Literal.String.Symbol]:
    result[i] = tokenizeRegex(result[i])
  elif statements[i] in[sqlparse.tokens.Name, sqlparse.tokens.Name.Placeholder, sqlparse.sql.Identifier]:
    old_value = result[i]
    if old_value in column_map:
      result[i] = column_map[old_value]
    else:
      result[i] = 'col'+ str(len(column_map))
      column_map[old_value] = result[i]
  elif (result[i] == "." and statements[i] == sqlparse.tokens.Punctuation and i > 0 and result[i-1].startswith('col')):
    old_value = result[i-1]
    if old_value in table_map:
      result[i-1] = table_map[old_value]
    else:
      result[i-1] = 'tab'+ str(len(table_map))
      table_map[old_value] = result[i-1]
  if (result[i].startswith('col') and i > 0 and (result[i-1] in ["from"])):
    old_value = result[i]
    if old_value in table_map:
      result[i] = table_map[old_value]
    else:
      result[i] = 'tab'+ str(len(table_map))
      table_map[old_value] = result[i]

tokenized_code = ' '.join(result)
print("SQL after tokenized: " + tokenized_code)

tokenized_code = "select time ( col0 ) from tab0"

# 3. make prediction
tokenizer=AutoTokenizer.from_pretrained("SEBIS/code_trans_t5_small_source_code_summarization_sql_multitask_finetune", skip_special_tokens=True)
ans = tokenizer(tokenized_code, return_tensors='pt')
#ans = pipeline([tokenized_code])
print(ans)


# direct encoding of the sample sentence
#tokenizer = AutoTokenizer.from_pretrained('SEBIS/code_trans_t5_small_source_code_summarization_sql_multitask_finetune')
#encoded_seq = tokenizer.encode(tokenized_code)

# your approach
# feature_extraction = pipeline('feature-extraction', tokenizer="SEBIS/code_trans_t5_small_source_code_summarization_sql_multitask_finetune")
feature_extraction = pipeline('feature-extraction', model="SEBIS/code_trans_t5_small_source_code_summarization_sql_multitask_finetune", tokenizer="SEBIS/code_trans_t5_small_source_code_summarization_sql_multitask_finetune")

features = feature_extraction(tokenized_code)
print(features.shape)
#print(features[0])
#print(encoded_seq)
# Compare lengths of outputs
#print(len(encoded_seq)) # 5
# Note that the output has a weird list output that requires to index with 0.
print(len(features[0])) # 5
"""