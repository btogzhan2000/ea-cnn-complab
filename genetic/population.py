import numpy as np
import hashlib
import copy
#from utils import StatusUpdateTool
import random

class Unit(object):
    def __init__(self, number):
        self.number = number


class ResUnit(Unit):
    def __init__(self, number, amount, in_channel, out_channel): #prob < 0.5
        super().__init__(number)
        self.type = 1
        self.amount = amount
        self.in_channel = in_channel
        self.out_channel = out_channel


class PoolUnit(Unit):
    def __init__(self, number, max_or_avg):
        super().__init__(number)
        self.type = 2
        self.max_or_avg = max_or_avg #max_pool for < 0.5 otherwise avg_pool


class DenseUnit(Unit):
    def __init__(self, number, amount, k, max_input_channel, in_channel, out_channel):
        super().__init__(number)
        self.type = 3
        self.amount = amount
        self.k = k
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.max_input_channel = max_input_channel

class InceptionBlock(Unit):	
    def __init__(self, number, in_channel, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool): 	
        super().__init__(number)	
        self.type = 4	
        # self.amount = amount	
        self.in_channel = in_channel	
        self.out_1x1 = out_1x1	
        self.red_3x3 = red_3x3	
        self.out_3x3 = out_3x3	
        self.red_5x5 = red_5x5	
        self.out_5x5 = out_5x5	
        self.out_1x1pool = out_1x1pool	
        self.inception_type = inception_type

class Individual(object):
    def __init__(self, params, indi_no):
        self.acc = -1.0
        self.id = indi_no # for record the id of current individual
        self.number_id = 0 # for record the latest number of basic unit
        self.max_len = params['max_len']
        self.image_channel = params['image_channel']
        self.output_channles = params['output_channel']

        self.min_resnet = params['min_resnet'] # minimal number of resnet units
        self.max_resnet = params['max_resnet'] # maximal number of resnet units
        self.min_pool = params['min_pool'] # minimal number of pool units
        self.max_pool = params['max_pool'] # maximal number of pool units
        self.min_densenet = params['min_densenet'] # minimal number of densenet units
        self.max_densenet = params['max_densenet'] # manimal number of densenet units
        self.min_inception = params['min_inception'] # minimal number of inception blocks
        self.max_inception = params['max_inception'] # maximal number of inception blocks

        self.min_resnet_unit = params['min_resnet_unit']
        self.max_resnet_unit = params['max_resnet_unit']

        self.k_list = params['k_list']
        self.min_k12 = params['min_k12'] # minimal number of k_12 for densenet
        self.max_k12 = params['max_k12']
        self.min_k20 = params['min_k20']
        self.max_k20 = params['max_k20']
        self.min_k40 = params['min_k40']
        self.max_k40 = params['max_k40']

        self.max_k12_input_channel = params['max_k12_input_channel'] # if the k is set to 12, its input channel cannot exceed this settings
        self.max_k20_input_channel = params['max_k20_input_channel']
        self.max_k40_input_channel = params['max_k40_input_channel']

        self.units = []

    def reset_acc(self):
        self.acc = -1.0

    def initialize(self):
        # initialize how many resnet unit/pooling layer/densenet unit will be used
        num_resnet = np.random.randint(self.min_resnet , self.max_resnet+1)
        num_pool = np.random.randint(self.min_pool , self.max_pool+1)
        num_densenet = np.random.randint(self.min_densenet, self.max_densenet+1)

        num_inception = np.random.randint(self.min_inception , self.max_inception+1)

        # find the position where the pooling layer can be connected
        #total_length = num_resnet + num_pool + num_densenet
        total_length = num_resnet + num_pool + num_densenet + num_inception

        all_positions = np.zeros(total_length, np.int32)
        if num_resnet > 0: all_positions[0:num_resnet] = 1;
        if num_pool > 0: all_positions[num_resnet:num_resnet+num_pool] = 2;
        if num_densenet > 0 : all_positions[num_resnet+num_pool:num_resnet+num_pool+num_densenet] = 3;

        if num_inception > 0: all_positions[num_resnet+num_pool+num_densenet: total_length] = 4;

        for _ in range(10):
            np.random.shuffle(all_positions)
        while all_positions[0] == 2: # pooling should not be the first unit
            np.random.shuffle(all_positions)

        # initialize the layers based on their positions
        input_channel = self.image_channel
        for i in all_positions:
            if i == 1:
                resnet = self.init_a_resnet(_number=None, _amount=None, _in_channel=input_channel, _out_channel=None)
                input_channel = resnet.out_channel
                self.units.append(resnet)
            elif i == 2:
                pool = self.init_a_pool(_number=None, _max_or_avg=None)
                self.units.append(pool)
            elif i == 3:
                densenet = self.init_a_densenet(_number=None, _amount=None, _k=None, _max_input_channel=None, _in_channel=input_channel)
                input_channel = densenet.out_channel
                self.units.append(densenet)
            elif i == 4:
          
                inception_types = ["3a", "3b", "4a", "4b", "4c", "4d", "4e", "5a", "5b"]
                chosen_type = random.choice(inception_types)
                # inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type="3a", out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1pool=32)

                if chosen_type == "3a":
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=64, red_3x3=96, out_3x3=128, red_5x5=16, out_5x5=32, out_1x1pool=32)
                elif chosen_type == "3b":
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=192, red_5x5=32, out_5x5=96, out_1x1pool=64)
                elif chosen_type == "4a":
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=192, red_3x3=96, out_3x3=208, red_5x5=16, out_5x5=48, out_1x1pool=64)
                elif chosen_type == "4b":
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=160, red_3x3=112, out_3x3=224, red_5x5=24, out_5x5=64, out_1x1pool=64)
                elif chosen_type == "4c":
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=128, red_3x3=128, out_3x3=256, red_5x5=24, out_5x5=64, out_1x1pool=64)
                elif chosen_type == "4d":
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=112, red_3x3=144, out_3x3=288, red_5x5=32, out_5x5=64, out_1x1pool=64)
                elif chosen_type == "4e":
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128)
                elif chosen_type == "5a":
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1pool=128)
                else:
                  inception = self.init_an_inception(_number=None, in_channels=input_channel, inception_type=chosen_type, out_1x1=384, red_3x3=192, out_3x3=384, red_5x5=48, out_5x5=128, out_1x1pool=128)
                
                # print("chosen type:", chosen_type)

                inception_out = inception.out_1x1 + inception.out_3x3 + inception.out_5x5 + inception.out_1x1pool
                input_channel = inception_out
                self.units.append(inception)
    """
    Initialize a resnet layer
    """
    def init_a_resnet(self, _number, _amount, _in_channel, _out_channel):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        if _amount:
            amount = _amount
        else:
            amount = np.random.randint(self.min_resnet_unit, self.max_resnet_unit+1)
        if _out_channel:
            out_channel = _out_channel
        else:
            out_channel = self.output_channles[np.random.randint(0, len(self.output_channles))]
        resnet = ResUnit(number, amount, _in_channel, out_channel)
        return resnet

    def init_a_pool(self, _number, _max_or_avg):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1

        if _max_or_avg:
            max_or_avg = _max_or_avg
        else:
            max_or_avg = np.random.rand()
        pool = PoolUnit(number, max_or_avg)
        return pool

    def init_a_densenet(self, _number, _amount, _k, _max_input_channel, _in_channel):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        if _k:
            k = _k;
        else:
            k = self.k_list[np.random.randint(0, len(self.k_list))]
        if _amount:
            amount = _amount
        else:
            amount_upper_limit = getattr(self, 'max_k%d'%(k))
            amount_lower_limit = getattr(self, 'min_k%d'%(k))
            amount = np.random.randint(amount_lower_limit, amount_upper_limit+1)
        if _max_input_channel:
            max_input_channel = _max_input_channel
        else:
            max_input_channel = getattr(self, 'max_k%d_input_channel'%(k))

        true_input = _in_channel
        densenet = DenseUnit(number, amount, k, max_input_channel, in_channel=_in_channel, out_channel=None)
        if true_input > densenet.max_input_channel:
            true_input = densenet.max_input_channel
        out_channel = true_input + k * amount
        densenet.out_channel = out_channel
        return densenet

    def init_an_inception(self, _number, in_channels, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        if _number:
            number = _number
        else:
            number = self.number_id
            self.number_id += 1
        # if _amount:
        #     amount = _amount
        # else:
        #     amount = np.random.randint(self.min_inception, self.max_inception+1)
        inception = InceptionBlock(number, in_channels, inception_type, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool)
        return inception

    def uuid(self):	
        _str = []	
        for unit in self.units:	
            _sub_str = []	
            if unit.type == 1:	
                _sub_str.append('resnet')	
                _sub_str.append('number:%d'%(unit.number))	
                _sub_str.append('amount:%d'%(unit.amount))	
                _sub_str.append('in:%d'%(unit.in_channel))	
                _sub_str.append('out:%d'%(unit.out_channel))	
            if unit.type == 2:	
                _sub_str.append('pool')	
                _sub_str.append('number:%d'%(unit.number))	
                _pool_type = 0.25 if unit.max_or_avg < 0.5 else 0.75	
                _sub_str.append('type:%.2f'%(_pool_type))	
            if unit.type == 3:	
                _sub_str.append('densenet')	
                _sub_str.append('number:%d'%(unit.number))	
                _sub_str.append('amount:%d'%(unit.amount))	
                true_in_channel = unit.in_channel	
                if true_in_channel > unit.max_input_channel:	
                    true_in_channel = unit.max_input_channel	
                _sub_str.append('in:%d'%(true_in_channel))	
                _sub_str.append('out:%d'%(unit.out_channel))	
            if unit.type == 4:	
                _sub_str.append('inception')	
                _sub_str.append('number:%d'%(unit.number))	
                # _sub_str.append('amount:%d'%(unit.amount))	
                _sub_str.append('in:%d'%(unit.in_channel))	
                _sub_str.append('out:%d'%(unit.out_1x1pool+unit.out_1x1+unit.out_3x3+unit.out_5x5))	
                _sub_str.append('type:%s'%(unit.inception_type))	
            _str.append('%s%s%s'%('[', ','.join(_sub_str), ']'))	
        _final_str_ = '-'.join(_str)	
        _final_utf8_str_= _final_str_.encode('utf-8')	
        _hash_key = hashlib.sha224(_final_utf8_str_).hexdigest()	
        return _hash_key, _final_str_	


    def __str__(self):
        _str = []
        _str.append('indi:%s'%(self.id))
        _str.append('Acc:%.5f'%(self.acc))
        for unit in self.units:
            _sub_str = []
            if unit.type == 1:
                _sub_str.append('resnet')
                _sub_str.append('number:%d'%(unit.number))
                _sub_str.append('amount:%d'%(unit.amount))
                _sub_str.append('in:%d'%(unit.in_channel))
                _sub_str.append('out:%d'%(unit.out_channel))

            if unit.type == 2:
                _sub_str.append('pool')
                _sub_str.append('number:%d'%(unit.number))
                _sub_str.append('type:%.1f'%(unit.max_or_avg))

            if unit.type == 3:
                _sub_str.append('densenet')
                _sub_str.append('number:%d'%(unit.number))
                _sub_str.append('amount:%d'%(unit.amount))
                _sub_str.append('k:%d'%(unit.k))
                _sub_str.append('in:%d'%(unit.in_channel))
                _sub_str.append('out:%d'%(unit.out_channel))
            if unit.type == 4:
                _sub_str.append('inception')
                _sub_str.append('number:%d'%(unit.number))
                # _sub_str.append('amount:%d'%(unit.amount))
                _sub_str.append('in:%d'%(unit.in_channel))
                _sub_str.append('out:%d'%(unit.out_1x1pool+unit.out_1x1+unit.out_3x3+unit.out_5x5))
                _sub_str.append('type:%s'%(unit.inception_type))
            
            _str.append('%s%s%s'%('[', ','.join(_sub_str), ']'))
        return '\n'.join(_str)

class Population(object):
    def __init__(self, params, gen_no):
        self.gen_no = gen_no
        self.number_id = 0 # for record how many individuals have been generated
        self.pop_size = params['pop_size']
        self.params = params
        self.individuals = []

    def initialize(self):
        for _ in range(self.pop_size):
            indi_no = 'indi%02d%02d'%(self.gen_no, self.number_id)
            self.number_id += 1
            indi = Individual(self.params, indi_no)
            indi.initialize()
            self.individuals.append(indi)

    def create_from_offspring(self, offsprings):
        for indi_ in offsprings:
            indi = copy.deepcopy(indi_)
            indi_no = 'indi%02d%02d'%(self.gen_no, self.number_id)
            indi.id = indi_no
            self.number_id += 1
            indi.number_id = len(indi.units)
            self.individuals.append(indi)


    def __str__(self):
        _str = []
        for ind in self.individuals:
            _str.append(str(ind))
            _str.append('-'*100)
        return '\n'.join(_str)






def test_individual(params):
    ind = Individual(params, 0)
    ind.initialize()
    print(ind)
    print(ind.uuid())

def test_population(params):
    pop = Population(params, 0)
    pop.initialize()
    print(pop)



# if __name__ == '__main__':
#     params = StatusUpdateTool.get_init_params()
#     test_individual(params)
#     test_population(params)
#     print("hello")





