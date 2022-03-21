#!/usr/bin/python3
import numpy as np
from generate_map_double import define_obstacle_space
import cv2
import random
import argparse
import math
import time
from queue import PriorityQueue

DEG_2_RAD = np.pi/180
class Robot:
    def __init__(self,x,y,theta,step_length):
        self.position_x = x
        self.position_y = y
        self.heading = theta
        self.step = step_length

    def update_position(self,x,y,theta):
        self.position_x = x
        self.position_y = y
        self.heading = theta

    def CheckMotion(self,theta):
        # print("calculating new position from :",(self.position_x,self.position_y))
        x = self.position_x + self.step * np.cos((theta+self.heading)*DEG_2_RAD)
        y = self.position_y + self.step * np.sin((theta+self.heading)*DEG_2_RAD)
        # print("x y before rounding:",x,y)

        x,y = round(x+1e-8),round(y+1e-8) 
        # print("x y after rounding:",x,y)

        th = (theta+self.heading)%360
        return (x,y,th)



class AStarSearch:
    def __init__(self,map_arr,start_location,goal_location, moveBindings,search_clearance):
        self.map_arr = map_arr
        self.goal_location = goal_location
        self.moveBindings = moveBindings
        self.start_location = start_location
        self.node_count = 1
        self.search_order = [300,330,0,30,60]
        self.clearance = search_clearance

        # self.visited_costs = np.ones((500,800,12))*np.inf
        initial_cost = get_cost_to_go((start_location[0],start_location[1]),goal_location)
        # self.visited_costs[start_location[0],start_location[1],start_location[2]//30] = initial_cost
        # Format: {(location_x,location_y,theta):[node_idx,parent_idx]}
        self.visited = {(start_location):[initial_cost, 0,None]}
        self.visited_costs = PriorityQueue()
        self.visited_costs.put((initial_cost,[start_location,start_location,0,None]))
        self.closed = {}

        # Format: {node_idx:[parent_node,(location)]}
        self.associations = {}

def assert_search_valid(search_state):
    print("checking search validity...")
    try:
        if(min(np.min(search_state.map_arr[search_state.start_location[0]-search_state.clearance:search_state.start_location[0]+search_state.clearance+1,search_state.start_location[1]-search_state.clearance:search_state.start_location[1]+search_state.clearance+1]),\
            np.min(search_state.map_arr[search_state.goal_location[0]-search_state.clearance:search_state.goal_location[0]+search_state.clearance+1,search_state.goal_location[1]-search_state.clearance:search_state.goal_location[1]+search_state.clearance+1])))== 0 :
            print("Obstacle/boundaries near start or end locations")
            return False
        else:
            print("Valid, searching now...")
            return True
    except Exception as e:
        # print(e)
        print("Obstacle/boundaries near start or end locations OR not in map_range")
        return False

def visualize(search_state,frame_skip):
    frame = 0
    last_location = None
    first_pass = True
    for location in search_state.closed:
        frame +=1
        search_state.map_arr[location[0],location[1]] = [0,255,0]
        # if not first_pass:
        #     cv2.line(search_state.map_arr,(location[1],location[0]),last_location,(0,255,0),1)
        last_location = (location[1],location[0])
        first_pass = False
        if frame % frame_skip ==0:
            cv2.imshow('map_arr',search_state.map_arr)
            cv2.waitKey(1)
    cv2.waitKey(0)
    backtrack(search_state)
    cv2.waitKey(0)

def backtrack(search_state):
    print("Backtracking now!")
    end_goal_location = list(search_state.closed.keys())[-1]
    cv2.circle(search_state.map_arr,(end_goal_location[1],end_goal_location[0]),5,(0,0,255),-1)
    search_state.map_arr[end_goal_location[0], end_goal_location[1]] = [0,0,255]
    end_goal_values = search_state.closed[end_goal_location]
    parent = end_goal_values[2]
    optimal_path = [end_goal_location]
    while(parent!=None):
        parent_location = search_state.associations[parent][1]
        optimal_path.append(parent_location)
        search_state.map_arr[parent_location[0], parent_location[1]] = [0,0,255]
        parent = search_state.associations[parent][0]

    cv2.circle(search_state.map_arr,(parent_location[1],parent_location[0]),5,(255,0,0),-1)
    print("optimal path length:",len(optimal_path))
    print("optimal_path:",optimal_path)
    cv2.imshow('map_arr',search_state.map_arr)


# Function which checks if a node is visited, if it has, returns the current cost, else returns -1
# def check_visited(visited_costs,location):
#     return visited_costs[location[0],location[1],location[2]//30]
def check_visited(visited,location):
    try:
        cost = visited[location][0]
    except:
        cost = None
    return cost

def check_closed(closed,location):
    try:
        _ = closed[location[0],location[1],location[2]]
        return True
    except:
        return False

def get_cost_to_go(curr_location,goal_location):
    cost = np.sqrt((curr_location[0] - goal_location[0])**2 + (curr_location[1] - goal_location[1])**2)
    return cost


def check_direction(search_state, robot, curr_node_location,curr_node_values,del_theta):
   
    # check if location is an obstacle or not, (x,y,theta)
    # print("current node location:",curr_node_location,"del theta:",del_theta)
    check_location = robot.CheckMotion(del_theta)
    # print("check location:",check_location)
    # print('check location for dtheta:',del_theta,"is: ",check_location)
    # check_location = tuple([sum(x) for x in zip(search_state.moveBindings[del_theta][0],curr_node_location)])
    if min(check_location[0]-search_state.clearance,check_location[0]+search_state.clearance+1,check_location[1]-search_state.clearance,check_location[1]+search_state.clearance+1) <0 :
        return
    if(np.min(search_state.map_arr[check_location[0]-search_state.clearance:check_location[0]+search_state.clearance+1,check_location[1]-search_state.clearance:check_location[1]+search_state.clearance+1]))== 0 :
        # print("obstacle found!! at ",check_location,"curr node locations:",curr_node_location)
        return
    
    if check_closed(search_state.closed,check_location):
        # print("'eliminimating repetition")
        return

    # if the location is never visited, append location and cost, returns either inf(not visited) or (C2C+C2G)
    check_res = check_visited(search_state.visited,check_location)
    # print("check_Res:",check_res)
    # if not check_closed(search_state.closed, check_location):
    # if check_location[0]==41 and check_location[1]==35:
        # print("CURR NODE Location:",curr_node_location, "check_location:",check_location,"del_theta:",del_theta,"node_count:",search_state.node_count)
    # print("check_Res:",check_res is np.inf)
    # if not visited
    if check_res is None:     # current cost                                                                      # moving cost  # cost to go
        # visit_cost = search_state.visited_costs[curr_node_location[0],curr_node_location[1],curr_node_location[2]//30] + 1 + get_cost_to_go((check_location[0],check_location[1]),search_state.goal_location)
        # search_state.visited_costs[check_location[0],check_location[1],check_location[2]//30] = visit_cost
        # print("adding node")
        search_state.visited[check_location] = [curr_node_values[0] + 5 + get_cost_to_go((check_location[0],check_location[1]),search_state.goal_location)-get_cost_to_go((curr_node_location[0],curr_node_location[1]),search_state.goal_location),\
                                                 search_state.node_count, curr_node_values[1]]
        search_state.visited_costs.put((curr_node_values[0] + 5 + get_cost_to_go((check_location[0],check_location[1]),search_state.goal_location)-get_cost_to_go((curr_node_location[0],curr_node_location[1]),search_state.goal_location),\
            [check_location,curr_node_values[0] + 5 + get_cost_to_go((check_location[0],check_location[1]),search_state.goal_location)-get_cost_to_go((curr_node_location[0],curr_node_location[1]),search_state.goal_location),\
                                                 search_state.node_count, curr_node_values[1]]))
        search_state.closed[check_location] = [curr_node_values[0] + 5 + get_cost_to_go((check_location[0],check_location[1]),search_state.goal_location)-get_cost_to_go((curr_node_location[0],curr_node_location[1]),search_state.goal_location),\
                                                 search_state.node_count, curr_node_values[1]]
        # file_nodes.write(str(check_location) + str([curr_node_values[0] + 5 + get_cost_to_go((check_location[0],check_location[1]),search_state.goal_location)-get_cost_to_go((curr_node_location[0],curr_node_location[1]),search_state.goal_location),\
                                                #  search_state.node_count, curr_node_values[1]]) + "\n")

        
        search_state.node_count+=1
    # if location has been visited, check if new cost is lower. If so, modify the node.
    elif check_res is not None:
        # print('check_res inside:',check_res is np.inf)
        if check_res > curr_node_values[0] + 5 + get_cost_to_go((check_location[0],check_location[1]),search_state.goal_location)-get_cost_to_go((curr_node_location[0],curr_node_location[1]),search_state.goal_location):
            print("updating existing cost")
            search_state.visited[check_location][0] = curr_node_values[0] + 5 + get_cost_to_go((check_location[0],check_location[1]),search_state.goal_location)-get_cost_to_go((curr_node_location[0],curr_node_location[1]),search_state.goal_location)
            search_state.visited[check_location][2] = curr_node_values[1]            
            search_state.closed[check_location][0] = curr_node_values[0] + 5 + get_cost_to_go((check_location[0],check_location[1]),search_state.goal_location)-get_cost_to_go((curr_node_location[0],curr_node_location[1]),search_state.goal_location)
            search_state.closed[check_location][2] = curr_node_values[1]            

# file_nodes = open("nodes.txt","w")
   
            

def astar_search(search_state, robot, visualize_search):
    curr_node_location = search_state.start_location
    curr_node_values = search_state.visited[curr_node_location]
    # curr_node_values = search_state.visited.get()

    map_copy = search_state.map_arr.copy()
    status_update_count = 0
    time_start = time.time()
    # while((curr_node_location[0],curr_node_location[1])!=search_state.goal_location): 
    # print("current distance:",get_cost_to_go(curr_node_location,search_state.goal_location))  
    while(get_cost_to_go(curr_node_location,search_state.goal_location)>10):  
        status_update_count+=1
        # if(curr_node_location)   
        # print("current distance from ",curr_node_location,":",get_cost_to_go(curr_node_location,search_state.goal_location))
        # Search all directions, if not visited, append, else modify only if lower cost is found
        for del_theta in search_state.search_order:
            check_direction(search_state, robot, curr_node_location,curr_node_values,del_theta)

        # Mark curr_location to closed nodes
        search_state.closed[curr_node_location] = curr_node_values
        
        # print("length of visited:",len(search_state.visited.keys()))
        # Save Parent-child relationship
        # print(search_state.visited.keys())
        search_state.associations[curr_node_values[1]] = [curr_node_values[2],curr_node_location]
        # Pop first visited element and sort for next iteration
        # search_state.visited.pop(next(iter(search_state.visited)))        
        # search_state.visited = {k: v for k, v in sorted(search_state.visited.items(), key=lambda item: item[1])}  # Sort visited based on cost
        search_state.visited_costs.get() # Pop
        next_vals = search_state.visited_costs.get()
        # print("next values:",next_vals)
        # print("next values0 location:",next_vals[1][0])
        # print("next values1:",next_vals[1])
        # print("next values:2",next_vals[2])


        curr_node_location = next_vals[1][0]


         # Set first node of visited as curr_Reached goal
        # curr_node_location = next(iter(search_state.visited))
        # print("exploring from node:",curr_node_location)
        # file_nodes.write(str(curr_node_location) + "\n")
        robot.update_position(curr_node_location[0],curr_node_location[1],curr_node_location[2])
        curr_node_values = [next_vals[1][1],next_vals[1][2],next_vals[1][3]]
        if(status_update_count==1000):
            time_end = time.time()
            print("time_taken for 1000 pops:",time_end - time_start)
            time_start = time.time()
            status_update_count = 0
            print("current_location:",curr_node_location, "current_distance:", get_cost_to_go((curr_node_location[0],curr_node_location[1]),search_state.goal_location))
        # map_copy[curr_node_location[0],curr_node_location[1]] = [0,255,0]
        # cv2.imshow('search',map_copy)
        # cv2.waitKey(1)

    print("final distance tolerance:",get_cost_to_go(curr_node_location,search_state.goal_location))
    print("final location:",curr_node_location)
    # Save goal locations and parent-child associations
    search_state.visited[curr_node_location]=curr_node_values
    search_state.closed[curr_node_location]=curr_node_values
    search_state.associations[curr_node_values[1]] = [curr_node_values[2],curr_node_location]

    # Visualization
    if visualize_search:
        visualize(search_state,frame_skip=50)



def main():
    map_arr = define_obstacle_space()
    # start_location = (random.randint(0,250)+1,random.randint(1,400)+1,0)
    # goal_location = (random.randint(0,250)+1,random.randint(1,400)+1,0)

    Parser = argparse.ArgumentParser()
    Parser.add_argument('--start_location_x', default="30", help='x coordinate of the start location, Default: 50')
    Parser.add_argument('--start_location_y', default="30", help='y coordinate of the start location, Default: 150')
    Parser.add_argument('--goal_location_x', default="200", help='x coordinate of the goal location, Default: 200')
    Parser.add_argument('--goal_location_y', default="200", help='y coordinate of the goal location, Default: 300')

    Args = Parser.parse_args()
    start_x = int(Args.start_location_x)
    start_y = int(Args.start_location_y)
    start_theta = 0
    goal_x = int(Args.goal_location_x)
    goal_y = int(Args.goal_location_y)


    search_clearance = 5
    start_location = (start_x,start_y,start_theta)
    goal_location = (goal_x,goal_y)
    print("start:",start_location)
    print("goal:",goal_location)
    
    visualize_search = True
    
    # Format: {Directions : [associated movement delta, costs]}
    moveBindings = {'e':[(0,1,0),1],'n':[(-1,0,0),1],'w':[(0,-1,0),1],'s':[(1,0,0),1],'ne':[(-1,1,0),1.4],\
                    'nw':[(-1,-1,0),1.4],'sw':[(1,-1,0),1.4],'se':[(1,1,0),1.4]}

    search_state = AStarSearch(map_arr,start_location,goal_location, moveBindings,search_clearance)

    if not(assert_search_valid(search_state)):
        cv2.circle(search_state.map_arr,(goal_location[1],goal_location[0]),search_clearance,(0,0,255),-1)
        cv2.circle(search_state.map_arr,(start_location[1],start_location[0]),search_clearance,(255,0,0),-1)
        cv2.imshow('map_arr',search_state.map_arr)
        cv2.waitKey(0)
        return
    robot = Robot(start_x,start_y,start_theta,step_length = 5)
    astar_search(search_state,robot,visualize_search)

# Priority Queue
if __name__=="__main__":
    # for testcase in range(50):
    #     cv2.destroyAllWindows()
    #     main()
    main()