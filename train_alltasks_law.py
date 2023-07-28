from torch.utils.data import DataLoader, TensorDataset
from data_loader.data_loader import *
import logging
from src.Distance import Q_compute, high_confidence
from src.model import Elastic_Tabular

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    handlers=[logging.FileHandler('./result/logging.txt')])



def datagenerator(data, label, protected, occupation, number):
    data = data[occupation == number]
    label = label[occupation == number]
    protected = protected[occupation == number]
    return data, label, protected,




if __name__ == '__main__':
    Taskname = 'adult'
    random_seed = 199719
    proportion = 0.7
    batch_size = 64
    data, label, protected, occupation = load_lawschool(random_seed)
    data_test, label_test, protected_test, occupation_test = load_lawschool_test(random_seed)

    data, data_test = torch.Tensor(data).to('cuda:0'), torch.Tensor(data_test).to('cuda:0')
    label, label_test = torch.Tensor(label).long().to('cuda:0'), torch.Tensor(label_test).long().to('cuda:0')
    protected, protected_test = torch.Tensor(protected).long().to('cuda:0'), torch.Tensor(protected_test).long().to('cuda:0')
    Q_list = []

    for epoch in range(1):
            for lambdavalue1 in [0.03,]:
                print('-----------------------')
                buffer = [ 2, 4, 5, 1, 0]
                knowledgebase, knowledgebase_label, knowledgebase_protected, = datagenerator(data, label, protected, occupation,3)


                for i in range(5):
                    batch_size_1 = int(knowledgebase.shape[0] / 150)
                    batch_size_2 = int(knowledgebase.shape[0] / 20)

                    train_data = (knowledgebase, knowledgebase_label, knowledgebase_protected,)
                    train_dataset = TensorDataset(*train_data)
                    train_loader = DataLoader(train_dataset, batch_size=batch_size_1, shuffle=True, drop_last=True )
                    columns = knowledgebase.size(1)
                    start_1 = Elastic_Tabular(train_loader, knowledgebase_label, knowledgebase_protected, columns,
                                              lambdavalue1,
                                              0, Taskname)

                    for _ in range(5):
                        start_1.Forward()


                    for number in buffer:

                        Q = Q_compute(knowledgebase, data[occupation == number],epoch=30, batch_size=batch_size_2)
                        Q_list.append(Q)

                    close_task = Q_list.index(max(Q_list)) #min for best case and max for worst


                    high_condifence_d, high_condifence_p = high_confidence(knowledgebase,
                                                                           data[occupation == buffer[close_task]],
                                                                           protected[occupation == buffer[close_task]],
                                                                           epoch=30, batch_size=batch_size_2)

                    high_condifence_y = start_1.high_confidence_label(high_condifence_d,)
                    torch.cuda.empty_cache()
                    print(len(high_condifence_y)/(len(label[occupation == buffer[close_task]])))

                    #start.Test(data_test, label_test, protected_test, data_test, )
                    knowledgebase = torch.cat((knowledgebase, high_condifence_d))
                    knowledgebase_label = torch.cat((knowledgebase_label, high_condifence_y))
                    knowledgebase_protected = torch.cat((knowledgebase_protected, high_condifence_p))

                    logging.info('The removed candidate tasks is task-{}'.format(buffer[close_task]))
                    print('The removed candidate tasks is task-{}'.format(buffer[close_task]))
                    if len(buffer) > 1:
                        buffer.remove(buffer[close_task])
                    Q_list = []

                batch_size = int(knowledgebase.shape[0] / 100)
                train_data = (knowledgebase, knowledgebase_label, knowledgebase_protected,)
                train_dataset = TensorDataset(*train_data)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)
                columns = knowledgebase.size(1)
                start_1 = Elastic_Tabular(train_loader, knowledgebase_label, knowledgebase_protected, columns,
                                          lambdavalue1,
                                          0, Taskname)
                for _ in range(5):
                    start_1.Forward()
                start_1.Test(data_test, label_test, protected_test, )
                for i in range(6):
                    start_1.Test(data_test[occupation_test == i], label_test[occupation_test == i], protected_test[occupation_test == i], )
                torch.cuda.empty_cache()

                logging.info('The removed candidate tasks is task-{}'.format(buffer[0]))
                print('The removed candidate tasks is task-{}'.format(buffer[0]))
                buffer.remove(buffer[0])



