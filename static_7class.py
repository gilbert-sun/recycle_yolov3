import os

def count_sub_pet_class(no,src1):

    global ct_all,fcount

    ct = []

    file_count = 0

    for fname in os.listdir(src1.format(no)):
        ff = os.path.join(src1.format(no) ,fname)
        with open(ff  , "r") as fp:
            for count in fp.readlines():
                ct.append(count[0])
                ct_all.append(count[0])
        file_count = file_count +1

    fcount = file_count + fcount

    print("\n---F:{}:--C:{}:---: 0):{} ,1):{} ,2):{} ,3):{} ,4):{} ,5):{} ,6):{} ! \n ".format(no,file_count, ct.count("0"),
                                                                                     ct.count("1"), ct.count("2"),
                                                                                     ct.count("3"), ct.count("4"),
                                                                                     ct.count("5"), ct.count("6")))

def count_total_pet_class(src_folder):
    # try:
        for no in range(0,len(os.listdir(src_folder))):
            if no >= 1:
              break
            src1 = os.path.join(src_folder,"labels")#src_folder + "{}/labels/"
            count_sub_pet_class(no,src1)
    # except:
    #     print ("---Err!\n")

if __name__ == '__main__':

    ct_all=[]

    fcount = 0

    src0 = "/media/e200/DATA/20190719_gilbert_full/all_in1/"

    count_total_pet_class(src0)
    sum = (lambda ct_all : (int(ct_all.count("0"))+ int(ct_all.count("1"))+ int(ct_all.count("2"))+ int(ct_all.count("3"))+ int(ct_all.count("4"))+ int(ct_all.count("5"))+ int(ct_all.count("6"))))
    # print (sum(ct_all))
    print("\nFinal---f_count:{}--:pet_total:{}--- P):{}-{}% ,O):{}-{}% ,S):{}-{}% ,C):{}-{}% ,Ot):{}-{}% ,T):{}-{}% ,Ch):{}-{}% ! \n ".format( fcount,
                                                                                      sum(ct_all),
                                                                                      ct_all.count("0"),{float(int(ct_all.count("0"))/sum(ct_all)*100)},
                                                                                      ct_all.count("1"),{float(int(ct_all.count("1"))/sum(ct_all)*100)},
                                                                                      ct_all.count("2"),{float(int(ct_all.count("2"))/sum(ct_all)*100)},
                                                                                      ct_all.count("3"),{float(int(ct_all.count("3"))/sum(ct_all)*100)},
                                                                                      ct_all.count("4"),{float(int(ct_all.count("4"))/sum(ct_all)*100)},
                                                                                      ct_all.count("5"),{float(int(ct_all.count("5"))/sum(ct_all)*100)},
                                                                                      ct_all.count("6"),{float(int(ct_all.count("6"))/sum(ct_all)*100)}))
