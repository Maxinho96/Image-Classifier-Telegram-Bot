from telegram.ext import Updater, MessageHandler, Filters
import os
import neural_network as nn

net = nn.NeuralNetwork()

def classify_image(bot, update):
    image_file = bot.getFile(update.message.photo[-1].file_id)
    image_file.download("image.jpg")
    preds = net.get_predictions("image.jpg")
    update.message.reply_text(preds)

def main():
    TOKEN = "634978695:AAHuQ47Fef5r0jnmKcbRZo8g38MoUo7DAus" #os.getenv("TOKEN")

    updater = Updater(TOKEN)
    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.photo, classify_image))

    updater.start_polling() #PORT = int(os.environ.get("PORT", "8443"))
    updater.idle() #HEROKU_APP_NAME = os.environ.get("HEROKU_APP_NAME")
    #updater.start_webhook(listen="0.0.0.0", port=PORT, url_path=TOKEN)
    #updater.bot.set_webhook("https://{}.herokuapp.com/{}".format(HEROKU_APP_NAME, TOKEN))


if __name__ == '__main__':
    main()
