
##-- Moving Average Filter --##

class MovingAverageFilter:
    
    def __init__(self):
        self.MAF_level = 8
        self.data = [0 for i in range(8)]
        print(self.data)

    def start(self, newData):
        self.data.pop()
        self.data.insert(0, newData)
        return (sum(self.data)/len(self.data))


# if __name__ == '__main__':

#     MovingAverageFilter = MovingAverageFilter(8)

#     x = [0, 0, 0, 0, 10, 20, 30, 35, 20, 10, 5, 0, 0, 0, 0]
#     for i in x:
#         print(MovingAverageFilter.start(i))
