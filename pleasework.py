import pandas as pd
import base64
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

# Load the train.csv file
train_df = pd.read_csv(r"C:\Users\wesle\Downloads\GettingHot\train.csv")
stemmer = PorterStemmer()
# Decode the Base64 encoded sentences
def stem_sentence(sentence):
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(stemmer.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


def decode_base64(data):
    try:
        decoded_data = base64.b64decode(data).decode('utf-8')
        return stem_sentence(decoded_data)
    except Exception:
        return None

train_df['decoded_sentence'] = train_df['sentence'].apply(decode_base64)

# Remove rows where decoding failed
train_df = train_df[train_df['decoded_sentence'].notna()]
train_df['stemmed_sentence'] = train_df['decoded_sentence'].apply(stem_sentence)

# Tokenize and vectorize the sentences
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train_df['stemmed_sentence'])

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, train_df['temperature'], test_size=0.2, random_state=42)

train_df['tokenized_sentence'] = train_df['stemmed_sentence'].apply(lambda x: tokenizer.encode(x, return_tensors='pt'))
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
X_train_tensor = torch.stack(X_train)
y_train_tensor = torch.tensor(y_train.values).float()

# Define the optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.MSELoss()

# Train the model
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = loss_fn(outputs[0], y_train_tensor)
    loss.backward()
    optimizer.step()
model.fit(X_train, y_train)
y_val_pred = model.predict(X_val)

# Compute the Mean Squared Error
mse = mean_squared_error(y_val, y_val_pred)

print(f'Mean Squared Error on validation set: {mse}')
# Load the test.csv file


test_df = pd.read_csv(r"C:\Users\wesle\Downloads\GettingHot\test.csv")

# Decode the Base64 encoded sentences in the test.csv file
test_df['decoded_sentence'] = test_df['sentence'].apply(decode_base64)

# Remove rows where decoding failed
test_df = test_df[test_df['decoded_sentence'].notna()]
test_df['stemmed_sentence'] = test_df['decoded_sentence'].apply(stem_sentence)
# Vectorize the test sentences
X_test = vectorizer.transform(test_df['stemmed_sentence'])

# Predict the temperatures for the test data
test_df['temperature'] = model.predict(X_test)

# Replace the '-1' placeholders in the submission.csv file with the predicted temperatures
submission_df = pd.read_csv(r"C:\Users\wesle\Downloads\GettingHot\submission.csv")
submission_df['temperature'] = test_df['temperature']

# Save the submission.csv file
submission_df.to_csv('newersubmission.csv', index=False)
