import io
import os

from telegram.ext import CommandHandler
from telegram.ext import Updater, MessageHandler, Filters

from model import get_prediction_by
from model.utils import get_date

telegram_bot_token = os.getenv("TOKEN")

updater = Updater(token=telegram_bot_token, use_context=True)
dispatcher = updater.dispatcher


def start(update, context):
    """
    Set up the introductory statement for the bot when the /start command is invoked
    """
    time = get_date()
    print(time + " Пользователь нажал кнопку start\n")
    chat_id = update.effective_chat.id
    greeting_message = "Приветствую! Приложите фотографию собаки, породу которой хотите классифицировать"
    message_note = "Пока я умею классифицировать только такие породы:\nShih-Tzu\nRhodesian ridgeback\nBeagle\n" \
                   "English foxhound\nBorder terrier\nAustralian terrier\nGolden retriever\nOld English sheepdog\n" \
                   "Samoyed\nDingo"
    context.bot.send_message(chat_id=chat_id,
                             text=greeting_message + "\n\n" + message_note)


def full_photo_input_handler(update, context):
    """
    Handle case when user upload document into bot
    """
    time = get_date()
    print(time + " Пользователь отправил документ")
    file = context.bot.get_file(update.message.document)
    file = io.BytesIO(file.download_as_bytearray())
    dog, prob = get_prediction_by(file)
    prob = "{:.1f}".format(prob * 100)
    message = f"Порода собаки: {dog}\nВероятность: {prob}%\n"
    print("Пользователь получил предсказание\n" + message)
    update.message.reply_text(message)


def compressed_photo_input_handler(update, context):
    """
    Handle case when user upload photo into bot
    """
    time = get_date()
    print(time + " Пользователь отправил фото")
    file = context.bot.get_file(update.message.photo[-1].file_id)
    file = io.BytesIO(file.download_as_bytearray())
    dog, prob = get_prediction_by(file)
    prob = "{:.1f}".format(prob * 100)
    message = f"Порода собаки: {dog}\nВероятность: {prob}%\n"
    print("Пользователь получил предсказание\n" + message)
    update.message.reply_text(message)


def text_input_handler(update, context):
    """
    Handle case when user text something to bot except for uploading photo
    """
    time = get_date()
    print(time + " Пользователь написал сообщение: " + update.message.text + "\n")
    message = "Я не понимаю слова, я понимаю только язык картинок с собаками, прикрепите, пожалуйста, фото собаки"
    update.message.reply_text(message)


# run the start function when the user invokes the /start command
dispatcher.add_handler(CommandHandler("start", start))

dispatcher.add_handler(MessageHandler(Filters.text, text_input_handler))
dispatcher.add_handler(MessageHandler(Filters.photo, compressed_photo_input_handler))
dispatcher.add_handler(MessageHandler(Filters.document, full_photo_input_handler))

updater.start_polling()
