import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load an image
img = mpimg.imread('home.jpg')

name = input("Enter your name: ")
print(f"Hi {name}! Welcome to the Jung Family's Long Drive Home!!\n")

# Display the image
imgplot = plt.imshow(img)
plt.show()

x = "s"
while x != "q":
    x = input("Type 1 if you want a story about Cora\nType 2\
if you want a story about Annabel\nType 3\
if you want a story about Mommy\nType 4\
if you want a story about Daddy\nType quit\
if you want to stop ")
    if x == "1":
        adj = input("Type an adjective: ")
        print(f"Once upon a time there was a {adj} girl named Cora.\n")
        adj2 = input("Type an adjective:")
        verb = input("Type a past-tense verb:")
        print(f"Cora has an older sister named Annabel.\nAnnabel is very {adj2}.\n\
        Cora rode in a car seat on the long drive home.\n\
        She napped and {verb} and chewed on her blanket.")
        v = input("Type a present-tense verb: ")
        print(f"Cora likes to {v}.\nBye friends, bye!")
    if x == "2":
        adj = input("Type an adjective: ")
        print(f"Once upon a time there was a girl named Annabel who was very {adj}.\n")
        adj2 = input("Type an adjective: ")
        verb = input("Type a past-tense verb:")
        print(f"Annabel had a doll named Abby which she brought on her long drive home.\n\
        Abby drank her {adj2} milk bottle every day.\n\
        On the long drive home, Annabel drew on her lucky clipboard, {verb}, and sang songs.")
    if x == "3":
        adj = input("Type an adjective: ")
        print(f"Once upon a time there was a lovely and {adj} mother named Jane.\n")
        adj2 = input("Type an adjective: ")
        print(f"Jane had a daughter named Annabel.\nAnnabel feels {adj2} today.\nThe end.")
    if x == "4":
        adj = input("Type an adjective: ")
        print(f"There was a father named Paul. Paul used to exercise when he felt {adj}\n")
        adj2 = input("Type an adjective: ")
        print(f"Paul had a daughter named Annabel.\nAnnabel feels {adj2} today.\nThe end.")