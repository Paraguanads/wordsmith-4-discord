# Main imports
import logging
import discord
import torch
from discord.ext import commands
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Old school logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)

# Load Google Firebase (not implemented atm)
import firebase_admin
from firebase_admin import firestore, credentials

cred = credentials.Certificate('path/to/your/credentials.json')
firebase_admin.initialize_app(cred)

db = firestore.client()

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

# Sending sentiment of N discord channel messages to Firebase (not implemented atm)
@bot.event
async def on_message(message):
    if message.channel.id == 'CHANNEL_DEV_ID':
        sentiment = sia.polarity_scores(message.content)
        data = {
            'user_id': message.author.id,
            'timestamp': message.created_at,
            'message': message.content,
            'sentiment': sentiment
        }
        db.collection('messages').add(data)

# Summarize a text invoked by /summarize yourtextgoeshere
def generate_summary(text, model, tokenizer, device, max_length=1024):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(device)
    summary_ids = model.generate(input_ids, max_length=max_length)
    summary_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary_text

@bot.command()
async def resume(ctx, *, text: str):
    try:
        summary = generate_summary(text, model, tokenizer, device)
        embed = discord.Embed(title='Summary', description=summary, color=0x00ff00)
        await ctx.send(embed=embed)
    except Exception as e:
        await ctx.send("Error: {}".format(e))

# Sentiment analysis command invoked by /sentiment your text goes here
@bot.command()
async def sentiment(ctx, *, text: str):
    sentiment = sia.polarity_scores(text)
    sentiment_text = sentiment['compound']
    sentiment_color = 0xffff00
    if sentiment_text > 0:
        sentiment_text = 'Positive'
        sentiment_color = 0x00ff00
    elif sentiment_text < 0:
        sentiment_text = 'Negative'
        sentiment_color = 0xff0000
    else:
        sentiment_text = 'Neutral'
    embed = discord.Embed(title='Predominant sentiment', description=sentiment_text, color=sentiment_color)
    embed.add_field(name='Negative', value=sentiment['neg'], inline=True)
    embed.add_field(name='Neutral', value=sentiment['neu'], inline=True)
    embed.add_field(name='Positive', value=sentiment['pos'], inline=True)
    await ctx.send(embed=embed)

# The bot is alive if you read this after running the script
@bot.event
async def on_ready():
    print(f'{bot.user} has connected to Discord!')

# Run the bot
bot.run('w3ir1)7ok3nfr0md!5(0rd)385i7e') # Discord bot token
