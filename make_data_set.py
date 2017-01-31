
import os
#act =['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell']
act =[ 'Bill Hader', 'Steve Carell']
all_images = os.listdir('cropped/')

train_set = open('training_set.txt','w+')
validation_set = open('validation_set.txt','w+')
testing_set = open('testing_set.txt','w+')
count = [0,0,0,0,0,0]
for image in all_images:
    i = 0
    for actor in act:
        if image[0:4] == actor.split()[1].lower()[0:4]:
            if count[i]<100:
                train_set.write(image+'\t'+actor.split()[1]+'\n')
            elif count[i]<110:
                validation_set.write(image+'\t'+actor.split()[1]+'\n')
            elif count[i]<120:
                testing_set.write(image+'\t'+actor.split()[1]+'\n')
            count[i] += 1
        i += 1
train_set.close()
validation_set.close()
testing_set.close()