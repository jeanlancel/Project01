import numpy as np
import matplotlib.pyplot as plt
import cPickle
from Linear_Regression import Data_Base, Linear_Classifier, make_data_set, Linear_Classifier_nD

imdb1 = Data_Base('training_set.txt','cropped/', {'Hader': -1, 'Carell': 1})
'''
classifier1 = Linear_Classifier(imdb1)
classifier1.grad_descent()
with open('linear_classifier1-2.pkl', 'wb') as f:
    cPickle.dump(classifier1,f,protocol=-1)

'''
with open('linear_classifier.pkl','rb') as f:
    classifier1 = cPickle.load(f)

#################### Validation Part 3###################################
imdb_val = Data_Base('validation_set.txt','cropped/', {'Hader': int(-1), 'Carell': int(1)})
classifier2 = Linear_Classifier(imdb_val)
classifier2.linear_weights = classifier1.linear_weights.copy()
output_val =classifier2.forward()
loss_val = classifier2.error(output_val)
print('Validation Loss: ')
print(loss_val)

pred_val = np.zeros(output_val.shape, dtype=int)
res = np.zeros(output_val.shape, dtype=int)
gt_label = classifier2.y
for i in range(0,output_val.shape[0]):
    if output_val[i]>=0:
        pred_val[i] = 1
    else:
        pred_val[i] = -1
    res[i] = pred_val[i]==int(gt_label[i])

accuracy = np.sum(np.array(res))/float(len(res))
print('Validation accuracy: ')
print(accuracy)
#Validation Loss:
#2.51565344979
#Validation accuracy:
#0.8

#################### Testing Part 3#######################################
imdb_test = Data_Base('testing_set.txt','cropped/', {'Hader': int(-1), 'Carell': int(1)})
classifier3 = Linear_Classifier(imdb_test)
classifier3.linear_weights = classifier1.linear_weights.copy()
output_val =classifier3.forward()
loss_val = classifier3.error(output_val)
print('Testing Loss: ')
print(loss_val)

pred_val = np.zeros(output_val.shape, dtype=int)
res = np.zeros(output_val.shape, dtype=int)
gt_label = classifier3.y
for i in range(0,output_val.shape[0]):
    if output_val[i]>=0:
        pred_val[i] = 1
    else:
        pred_val[i] = -1
    res[i] = pred_val[i]==int(gt_label[i])

accuracy = np.sum(np.array(res))/float(len(res))
print('Testing accuracy: ')
print(accuracy)
#Validation Loss:
#0.252029348837
#Validation accuracy:
#0.85
#Testing Loss:
#0.188357613313
#Testing accuracy:
#0.95

#################### Part 4 #######################################
heatmap = classifier1.linear_weights[1:].reshape((32,32))
plt.imsave('Image_outputs/200image_trained.png',heatmap)
hmap = plt.imshow(heatmap, cmap='gray', interpolation= 'nearest')
#plt.show(hmap)

errors = classifier1.f_t
iters = np.linspace(0,errors.size-1,errors.size)
error_descent = plt.plot(iters,errors)
plt.ylabel('Error')
plt.xlabel('Iteration')
plt.show(error_descent)

with open('4imgset.txt','wb') as f:
    f.write('carell44.jpg\tCarell\n'
            'carell39.jpg\tCarell\n'
            'hader48.jpg\tHader\n'
            'hader132.jpg\tHader\n')

imdb4 = Data_Base('4imgset.txt','cropped/',{'Hader': int(-1), 'Carell': int(1)})
classifier4 = Linear_Classifier(imdb4)
classifier4.grad_descent()

with open('linear_classifier4.pkl', 'wb') as f:
    cPickle.dump(classifier4,f,protocol=-1)

heatmap = classifier4.linear_weights[1:].reshape((32,32))
plt.imsave('Image_outputs/4_image_trained.png',heatmap)
hmap2 = plt.imshow(heatmap,cmap='gray', interpolation= 'None')
#plt.show(hmap2)

#################### Part 5 #######################################
make_data_set(['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'],'5a',140)
make_data_set(['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'],'5b',110)
make_data_set(['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'],'5c',80)
make_data_set(['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'],'5d',50)
make_data_set(['Fran Drescher', 'America Ferrera', 'Kristin Chenoweth', 'Alec Baldwin', 'Bill Hader', 'Steve Carell'],'5e',20)
make_data_set(['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan', 'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon'],'5t',20)
testing_data = Data_Base('training_set5t.txt','cropped/',{'Butler':-1, 'Radcliffe':-1,
                        'Vartan':-1, 'Bracco':1, 'Gilpin':1, 'Harmon':1})
val_data5 = Data_Base('training_set5a.txt','cropped/',{'Drescher':1,
                                                  'Ferrera':1,
                                                  'Chenoweth':1,
                                                  'Baldwin':-1,
                                                  'Hader':-1,
                                                  'Carell':-1})

ids = ['5a','5b','5c','5d','5e']
dbs = []
for id in ids:
    dbs.append(Data_Base('training_set%s.txt'%id,'cropped/',{'Drescher':1,
                                                  'Ferrera':1,
                                                  'Chenoweth':1,
                                                  'Baldwin':-1,
                                                  'Hader':-1,
                                                  'Carell':-1}))
i=0
training_res = np.ndarray(5)

#run regression
for db in dbs:
    #classifier5 = Linear_Classifier(db)
    #classifier5.grad_descent()
    #with open('linear_classifier%s-2.pkl'%ids[i], 'wb') as f:
    #    cPickle.dump(classifier5,f,protocol=-1)

    with open('linear_classifier%s-2.pkl'%ids[i], 'rb') as f:
        classifier5 = cPickle.load(f)
    output_training = classifier5.forward()
    pred = np.zeros(output_training.shape, dtype=int)

    res = np.zeros(output_training.shape, dtype=int)
    gt_label = classifier5.y

    for j in range(0, output_training.shape[0]):
        if output_training[j] >= 0:
            pred[j] = 1
        else:
            pred[j] = -1
        res[j] = pred[j] == int(gt_label[j])

    accuracy = np.sum(np.array(res)) / float(len(res))
    print('Training accuracy: ')
    print(accuracy)
    training_res[i] = accuracy
    i+=1


    plt.plot([140, 110, 80, 50, 20], training_res)
    plt.ylim([0.9,1])
    plt.xlabel('Number of Image per actor')
    plt.ylabel('Accuracy on training set')
    #plt.show()
    plt.savefig('Image_outputs/training_res_part5.png')

#test classifiers
testing_results = np.ndarray((5,2))
j = 0
for id in ids:
    imdb_test = testing_data
    imdb_val = val_data5
    classifier5t = Linear_Classifier(imdb_test)
    classifier5v = Linear_Classifier(imdb_val)
    with open('linear_classifier%s-2.pkl'%id, 'rb') as f:
        classifier5o = cPickle.load(f)
    classifier5t.linear_weights = classifier5o.linear_weights.copy()
    classifier5v.linear_weights = classifier5o.linear_weights.copy()
    output_val = classifier5v.forward()
    output_test = classifier5t.forward()
    vloss_val = classifier5v.error(output_val)
    loss_val = classifier5t.error(output_test)
    print('Validation Loss')
    print(vloss_val)
    print('Testing Loss: ')
    print(loss_val)


    pred_val = np.zeros(output_val.shape, dtype=int)
    res = np.zeros(output_val.shape, dtype=int)
    gt_label = classifier5v.y
    for i in range(0, output_val.shape[0]):
        if output_val[i] >= 0:
            pred_val[i] = 1
        else:
            pred_val[i] = -1
        res[i] = pred_val[i] == int(gt_label[i])

    accuracyV = np.sum(np.array(res)) / float(len(res))
    print('Validation accuracy: ')
    print(accuracyV)

    pred_val = np.zeros(output_test.shape, dtype=int)
    res = np.zeros(output_test.shape, dtype=int)
    gt_label = classifier5t.y
    for i in range(0, output_test.shape[0]):
        if output_test[i] >= 0:
            pred_val[i] = 1
        else:
            pred_val[i] = -1
        res[i] = pred_val[i] == int(gt_label[i])

    accuracy = np.sum(np.array(res)) / float(len(res))
    print('Testing accuracy: ')
    print(accuracy)
    testing_results[j][0] = accuracyV
    testing_results[j][1] = accuracy
    j+=1

plt.plot([140, 110, 80, 50, 20], testing_results[:,0])
plt.plot([140, 110, 80, 50, 20], testing_results[:,1])
plt.xlabel('Number of Image per actor')
plt.ylabel('Accuracy')
plt.legend(['Validation','Testing'])
plt.show()
'''
Validation Loss
0.0556209862341
Testing Loss:
0.301930647316
Validation accuracy:
0.994047619048
Testing accuracy:
0.816666666667
Validation Loss
0.0747351836277
Testing Loss:
0.318667847575
Validation accuracy:
0.977380952381
Testing accuracy:
0.8
Validation Loss
0.103783873899
Testing Loss:
0.322492168105
Validation accuracy:
0.958333333333
Testing accuracy:
0.783333333333
Validation Loss
0.150479615249
Testing Loss:
0.417336222088
Validation accuracy:
0.929761904762
Testing accuracy:
0.683333333333
Validation Loss
0.21055522192
Testing Loss:
0.396036591325
Validation accuracy:
0.886904761905
Testing accuracy:
0.733333333333
'''

#################### Part 6 #######################################

#create new set of data
imdb6 = Data_Base('training_set5a.txt','cropped/',
                 {'Drescher':[1,0,0,0,0,0], 'Ferrera':[0,1,0,0,0,0], 'Chenoweth':[0,0,1,0,0,0],
                  'Baldwin':[0,0,0,1,0,0], 'Hader':[0,0,0,0,1,0], 'Carell':[0,0,0,0,0,1]})
#initialize classifier for n class classification
classifier6 = Linear_Classifier_nD(imdb6)
#obtain initial gradient
cl_grad = classifier6.grad(np.dot(classifier6.x, classifier6.linear_weights))
#random set of points to test for gradient computation
points = [[107,1],[0,0],[420,4],[329,2],[720,3],[185,5],[262,4],[638,0]]
#print out finite difference derivative alongside corresponding values of Grad
for point in points:
    h = np.zeros((1025, 6))
    h[point[0],point[1]] = 0.001
    error_a = classifier6.error(np.dot(classifier6.x, classifier6.linear_weights + h))
    error_b = classifier6.error(np.dot(classifier6.x, classifier6.linear_weights))
    true_deri = (error_a - error_b) / 0.001
    print "Manual Derivative: %f \tFrom Gradient: %f" %(true_deri, cl_grad[point[0],point[1]])

#################### Part 7 #######################################
#Perform gradient descent and save file for future use
classifier6.grad_descent()
'''
with open('classifier6.pkl','wb') as f:
    cPickle.dump(classifier6,f,protocol = -1)
'''
with open('classifier6.pkl','rb') as f:
    classifier6 = cPickle.load(f)

weights = classifier6.linear_weights.copy()

#Validation
val_data7 = Data_Base('validation_set5a.txt','cropped/',{'Drescher':[1,0,0,0,0,0], 'Ferrera':[0,1,0,0,0,0], 'Chenoweth':[0,0,1,0,0,0],
                  'Baldwin':[0,0,0,1,0,0], 'Hader':[0,0,0,0,1,0], 'Carell':[0,0,0,0,0,1]})

classifier7_val = Linear_Classifier_nD(val_data7)
classifier7_val.linear_weights = weights.copy()
outputs_val7 = classifier7_val.forward()
gt_label = classifier7_val.y

pred_val = np.zeros(outputs_val7.shape, dtype=int)
res = np.zeros(outputs_val7.shape[0], dtype=int)
for i in range(0,pred_val.shape[0]):
    pred_val[i,np.argmax(outputs_val7[i])] = 1
    res[i] = int(np.sum(pred_val[i]-gt_label[i]))==0

accuracy = np.sum(np.array(res)) / float(len(res))
print('Validation accuracy: ')
print(accuracy)

test_data7 = Data_Base('testing_set5a.txt','cropped/',{'Drescher':[1,0,0,0,0,0], 'Ferrera':[0,1,0,0,0,0], 'Chenoweth':[0,0,1,0,0,0],
                  'Baldwin':[0,0,0,1,0,0], 'Hader':[0,0,0,0,1,0], 'Carell':[0,0,0,0,0,1]})
classifier7_test = Linear_Classifier_nD(test_data7)
classifier7_test.linear_weights = weights.copy()
outputs_val7 = classifier7_test.forward()
gt_label = classifier7_test.y

pred_val = np.zeros(outputs_val7.shape, dtype=int)
res = np.zeros(outputs_val7.shape[0], dtype=int)
for i in range(0,pred_val.shape[0]):
    pred_val[i,np.argmax(outputs_val7[i])] = 1
    res[i] = int(np.sum(pred_val[i]-gt_label[i]))==0

accuracy = np.sum(np.array(res)) / float(len(res))
print('Testing Accuracy: ')
print(accuracy)

for j in range(0,6):
    heatmap = weights[1:,j].copy().reshape((32, 32))
    heatmap /=heatmap.max()
    heatmap +=1.
    #heatmap = plt.imshow(heatmap, cmap='heat', interpolation='None')
    #plt.show()
    hmap7 = plt.imsave('Image_outputs/actor_%i_of_6.png'%(j+1),heatmap,cmap='coolwarm')



