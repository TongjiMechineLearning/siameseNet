import urllib
import  os
import urllib.request
import time
import shutil

def listdir(path, list_name):  # 传入存储的list
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)
    return list_name

def write():
    rootdir = "/home/aiserver/dataset/imretrieval/大花叠在一起"
    folder_data = ["/home/aiserver/code/SiameseNet/data"]
    list_name = []
    for path in folder_data:
        list_name = listdir(path, list_name)

    targetPath = "/home/aiserver/IdeaProjects/demo_retrieval_v1/target/demo/indexer_images"


    #list = os.listdir(rootdir)

    for i in range(0, len(list_name)):

        path = os.path.join(rootdir, list_name[i])

        if os.path.isfile(path):
            name = path.split("/")[-1]
            root_name = "{}_{}.jpg".format(name.split(".")[0], i)
            shutil.copy(path, "{}/{}".format(targetPath, root_name))
            print(name)
            url = "http://localhost:8080/writeToIndex?rootpath="+root_name +"&realpath="+root_name
            print(i, "----", url)
            response = urllib.request.urlopen(url)
            print(response.read())
            #time.sleep(1)

def create_index():
    url = ""


if __name__ == '__main__':
    write()
