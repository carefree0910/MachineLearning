import pickle
import random


def dic():
    new_data = yield
    word, tp, accent, meaning = new_data["word"], new_data["type"], new_data["accent"], new_data["meaning"]
    data_path = new_data["path"]
    if not word:
        print("Please don't send an empty word to the dictionary!")
    else:
        with open(data_path + "dic.dat", "rb") as file:
            pack = pickle.load(file)

            pack["dictionary"][(word, tp)] = {
                "accent": accent,
                "meaning": meaning
            }

            new_data = {
                "word": word,
                "type": tp,
                "accent": accent,
                "meaning": meaning
            }

            succeed = {
                "suc": True
            }
            failed = {
                "suc": False,
                "old": None,
                "new": new_data
            }

            for i, data in enumerate(pack["list"]):
                if data["word"] == word and data["type"] == tp:
                    failed["old"] = data
                    respond = yield failed
                    if respond["modify"]:
                        length = len(pack["list"])
                        pack["list"][i] = respond["data"]
                        pack["list"][length - 1], pack["list"][i] = pack["list"][i], pack["list"][length - 1]
                    break
            else:
                yield succeed
                pack["list"].append(new_data)

        with open(data_path + "dic.dat", "wb") as file:
            pickle.dump(pack, file)


def print_dic(data_path):
    with open(data_path + "dic.dat", "rb") as file:
        pack = pickle.load(file)
        print(pack["list"])
        print(pack["dictionary"])


def reset_dic(data_path):
    with open(data_path + "dic.dat", "wb") as file:
        pickle.dump({
            "dictionary": {},
            "list": []
        }, file)


def shuffle_dic(data_path):
    with open(data_path + "dic.dat", "rb") as file:
        pack = pickle.load(file)
        lst, dictionary = pack["list"], pack["dictionary"]
        random.shuffle(lst)
    with open(data_path + "dic.dat", "wb") as file:
        pickle.dump({
            "dictionary": dictionary,
            "list": lst
        }, file)


def rotate_list(idx, data_path):
    with open(data_path + "dic.dat", "rb") as file:
        pack = pickle.load(file)
        lst, dictionary = pack["list"], pack["dictionary"]
        selected = lst.pop(idx)
        lst.append(selected)
    with open(data_path + "dic.dat", "wb") as file:
        pickle.dump({
            "dictionary": dictionary,
            "list": lst
        }, file)


def find_word(word, data_path):
    with open(data_path + "dic.dat", "rb") as file:
        dictionary = pickle.load(file)["dictionary"]
        data_list = []
        for key, data in dictionary.items():
            if word in key[0]:
                data_list.append({
                    "word": key[0],
                    "type": key[1],
                    "accent": data["accent"],
                    "meaning": data["meaning"]
                })
        return data_list


def delete_word(word, tp, data_path):
    if not word:
        print("Please delete a valid word!")
    else:
        with open(data_path + "dic.dat", "rb") as file:
            pack = pickle.load(file)
            pack["dictionary"].pop((word, tp), None)
            for i, data in enumerate(pack["list"]):
                if data["word"] == word and data["type"] == tp:
                    pack["list"].pop(i)
                    break
        with open(data_path + "dic.dat", "wb") as file:
            pickle.dump(pack, file)
