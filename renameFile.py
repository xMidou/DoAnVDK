import pathlib


def rename_photos():
    path = pathlib.Path('.')/"test/2"
    count  = 102
    for folder in path.iterdir():
        folder.rename("User.2." + str(count) + ".jpg")
        count += 1
        print(count)

if __name__== "__main__":
     rename_photos()
    
