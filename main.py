# Main imports
import logging
import discord
import torch
from discord.ext import commands
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Old school logging. You can turn this off on production.
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

# Load BART tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

# Discord mandatory intents and command prefix setting
intents = discord.Intents.all()
bot = commands.Bot(command_prefix='/', intents=intents)

# Function to predict sentiment on text
def predict_sentiment(text, model, tokenizer, device):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    # Add a batch dimension to the input tensor
    input_ids = input_ids.unsqueeze(0)
    logits = model(input_ids)[0]
    # Remove the batch dimension from the logits tensor
    logits = logits.squeeze(0)
    # Convert logits to a Python scalar
    logits = logits.tolist()[0]
    if logits > 0:
        return "positive"
    else:
        return "negative"

# Sentiment analysis
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Summarize a text. Invoked by /summarize your text goes here
def generate_summary(text, model, tokenizer, device, max_length=1024):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    summary_ids = model.generate(input_ids, max_length=max_length)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

@bot.command()
async def resume(ctx, *, text: str):
    try:
        summary = generate_summary(text, model, tokenizer, device)
        await ctx.send("Summary: {}".format(summary))
    except Exception as e:
        await ctx.send("Error: {}".format(e))

# Sentiment analysis command. Invoked by /sentiment your text goes here
@bot.command()
async def sentiment(ctx, *, text: str):
    try:
        sentiment = sia.polarity_scores(text)
        await ctx.send("Sentiments on the provided text are between positive ~ {0},"
                       " negative ~ {1} and neutral ~ {2}".format(sentiment['pos'],
                                                           sentiment['neg'],
                                                           sentiment['neu']))
    except Exception as e:
        await ctx.send("Error: {}".format(e))

# The bot is alive if you read this after running the script
@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

# Error if the user doesn't provide a valid command
@bot.event
async def on_command_error(ctx, error):
    if isinstance(error, commands.CommandNotFound):
        await ctx.send("Error: The command doesn't exist")
    else:
        await ctx.send("Error: {}".format(error))

# Run the bot
bot.run('w3ir1)7ok3nfr0md!5(0rd)385i7e') # Discord bot token