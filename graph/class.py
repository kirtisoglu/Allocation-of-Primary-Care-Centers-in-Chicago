

class Node():
    def __init__(self, name, ID, x_axis, y_axis, population, distance, neighbors, EPHC, PPHC):
        self.name = name
        self.ID = ID
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.population = population
        self.distance = distance
        self.neighbors = neighbors
        self.EPHC = EPHC
        self.PPHC = PPHC

    def get_node_distance(self):
        return self.distance

    def get_node_population(self):
        return self.population       
    
    def get_node_ID(self):
        return self.ID

    def get_node_neighbors(self):
        return self.neighbors

    def get_node_name(self):
        return self.name

    def get_node_x_axis(self):
        return self.x_axis

    def get_node_y_axis(self):
        return self.y_axis

    def get_node_position(self):
        return self.x_axis, self.y_axis    

    def get_node_EPHC(self):
        return self.EPHC 

    def get_node_PPHC(self):
        return self.PPHC

#n1 = Node(1,2,10,[1,2,3,4],[])
# print(n1.get_position())