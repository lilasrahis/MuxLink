#!/usr/bin/python
import copy
import os
import sys
import numpy as np
import re
import code
import tempfile
import networkx as nx
import random

## Here, we implement the DMUX scheme to the best of our ability based on the description in the TCAD paper
benchmark=sys.argv[1]
key_size=int(sys.argv[2])
location=sys.argv[3]
print("Benchmark is "+benchmark)
print("Key-size is "+str(key_size))

def GenerateKey(K):
    nums= np.ones(K)
    nums[0:(K//2)] =0
    np.random.shuffle(nums)
    return nums

def parse(path, dump=False):
    top = path.split('/')[-1].replace('.bench','')
    with open(path, 'r') as f:
        data = f.read()
    return verilog2gates(data, dump)
def ExtractSingleMultiOutputNodes(G):
    F_multi=[]
    F_single=[]
    for n in G.nodes():
        out_degree=c.out_degree(n)
        check=c.nodes[n]['output']
        if out_degree == 1:
            if c.nodes[n]['gate'] != "input" and not check:
                F_single.append(n)
        elif out_degree>1:
            if c.nodes[n]['gate'] != "input" and not check:
                F_multi.append(n)
        else:
            if not check:
                print("Node "+n+" has 0 output and it is not an output")
    return F_multi, F_single


def verilog2gates(verilog, dump):
    Dict_gates={'xor':[0,1,0,0,0,0,0,0],
    'XOR':[0,1,0,0,0,0,0,0],
    'OR':[0,0,1,0,0,0,0,0],
    'or':[0,0,1,0,0,0,0,0],
    'XNOR':[0,0,0,1,0,0,0,0],
    'xnor':[0,0,0,1,0,0,0,0],
    'and':[0,0,0,0,1,0,0,0],
    'AND':[0,0,0,0,1,0,0,0],
    'nand':[0,0,0,0,0,1,0,0],
    'buf':[0,0,0,0,0,0,0,1],
    'BUF':[0,0,0,0,0,0,0,1],
    'NAND':[0,0,0,0,0,1,0,0],
    'not':[0,0,0,0,0,0,1,0],
    'NOT':[0,0,0,0,0,0,1,0],
    'nor':[1,0,0,0,0,0,0,0],
    'NOR':[1,0,0,0,0,0,0,0],
}
    G = nx.DiGraph()
    ML_count=0
    regex= "\s*(\S+)\s*=\s*(BUF|NOT)\((\S+)\)\s*"
    for output, function, net_str in re.findall(regex,verilog,flags=re.I |re.DOTALL):
        input=net_str.replace(" ","")


        G.add_edge(input,output)
        G.nodes[output]['gate'] = function
        G.nodes[output]['count'] = ML_count
        if dump:

            feat=Dict_gates[function]
            for item in feat:
                f_feat.write(str(item)+" ")
            f_feat.write("\n")
            f_cell.write(str(ML_count)+" assign for output "+output+"\n")
            f_count.write(str(ML_count)+"\n")
        ML_count+=1
    regex= "(\S+)\s*=\s*(OR|XOR|AND|NAND|XNOR|NOR)\((.+?)\)\s*"
    for output, function, net_str in re.findall(regex,verilog,flags=re.I | re.DOTALL):
        nets = net_str.replace(" ","").replace("\n","").replace("\t","").split(",")
        inputs = nets
        G.add_edges_from((net,output) for net in inputs)
        G.nodes[output]['gate'] = function
        G.nodes[output]['count'] = ML_count
        if dump:
            feat=Dict_gates[function]
            for item in feat:
                f_feat.write(str(item)+" ")
            f_feat.write("\n")
            f_cell.write(str(ML_count)+" assign for output "+output+"\n")
            f_count.write(str(ML_count)+"\n")
        ML_count+=1
    for n in G.nodes():
        if 'gate' not in G.nodes[n]:
            G.nodes[n]['gate'] = 'input'
    for n in G.nodes:
        G.nodes[n]['output'] = False
    out_regex = "OUTPUT\((.+?)\)\n"
    for net_str in re.findall(out_regex,verilog,flags= re.I | re.DOTALL):
        nets = net_str.replace(" ","").replace("\n","").replace("\t","").split(",")
        for net in nets:
            if net not in G:
                print("Output "+net+" is Float")
            else:
                G.nodes[net]['output'] = True
    if dump:
        for u,v in G.edges:
            if 'count' in G.nodes[u].keys() and 'count' in G.nodes[v].keys():
                f_link_train.write(str(G.nodes[u]['count'])+" "+str(G.nodes[v]['count'])+"\n")
    return G



def FindPairs(S_sel, F_single, F_multi, I_max, O_max, G, selected_g):

    F1=[]
    F2=[]
    if S_sel == "s1" or S_sel =="s2":
        F1=F_multi
        F2=F_multi
    elif S_sel == "s3":
        F1=F_multi
        F2=F_single
    else:
        F1=F_multi+F_single
        F2=F_multi+F_single
    done = False
    i=0
    f1=""
    f2=""
    g1=""
    g2=""
    while i < I_max:
        f1=random.choice(F1)
        f2 = f1
        while f2 == f1:
            f2=random.choice(F2)

        j=0
        while j< O_max:
            g1=random.choice(list(G.successors(f1)))
            g2=random.choice(list(G.successors(f2)))
            R1= nx.has_path(G,g1,f2)
            R2= nx.has_path(G,g2,f1)
            if (g1 != g2) and not R1 and not R2:
                if g1 not in selected_g and g2 not in selected_g:
                    done = True
                    break
            j=j+1
        if done:
            break
        i=i+1
    if done:
        if S_sel == "s1" or S_sel =="s4":
            G.add_edge(f2,g1)
            G.add_edge(f1,g2)
        else:
            G.add_edge(f2,g1)
    return f1, f2, g1, g2, done, G
if __name__=='__main__':
    if not os.path.exists(location):
        os.mkdir(location)
        print("Directory '% s' created" % location)
    f_feat = open(location+"/feat.txt", "w")
    f_cell = open(location+"/cell.txt", "w")
    f_count = open(location+"/count.txt", "w")
    f_link_test = open(location+"/links_test.txt", "w")
    f_link_train = open(location+"/links_train_temp.txt", "w")
    f_link_train_f = open(location+"/links_train.txt", "w")
    f_link_test_neg = open(location+"/link_test_n.txt", "w")
    c = parse('./Benchmarks/'+benchmark+'.bench', True)
    new_c = parse('./Benchmarks/'+benchmark+'.bench')
    F_multi, F_single = ExtractSingleMultiOutputNodes(c)
    #Generate the key
    K_list= GenerateKey(key_size)
    counter=key_size-1
    myDict = {}
    #choices for locking. s4 is avilable always. So it is only used when needed
    L_s= np.array(["s1", "s2", "s3"])
    selected_f1_gates=[]
    selected_f2_gates=[]
    selected_g2_gates=[]
    selected_g1_gates=[]
    selected_g=[]
    break_out=0
    while counter>=0:
        print("key search counter is "+str(counter))

        print("F single size is "+str(len(F_single)))
        print("F multi size is "+str(len(F_multi)))
        np.random.shuffle(L_s)
        fallback=True
        S_sel=""
        for s in L_s:
            if s =="s1" and counter<2:
                print("Choice is s1 but counter is less than 2")
                continue
            elif s=="s3" and len(F_multi)>1 and len(F_single)>1:
                S_sel=s
                fallback=False
                break
            elif len(F_multi)<2:

                print("Choice is "+s+" Length of multi is less than 2")
                continue
            S_sel=s
            fallback=False
            break
        if fallback:
            S_sel="s4"
        print("Chosen is "+S_sel)
        to_be_new_c = nx.DiGraph()
        done = False
        f1, f2, g1, g2, done, to_be_new_c = FindPairs(S_sel, F_single, F_multi, 10, 10, new_c, selected_g)
        if not done:
            break_out+=1
            if (break_out>=10):
                print("Tried 10 times")
                break_out=0;
                S_sel="s4"
                while not done:
                    print("Calling again with s4")
                    f1, f2, g1, g2, done, to_be_new_c = FindPairs("s4", F_single, F_multi, 10, 10, new_c, selected_g)
            else:
                continue
        if len(list(nx.simple_cycles(to_be_new_c))) !=0:
            print("There is a cycle")
            continue
        new_c = copy.deepcopy(to_be_new_c)
        selected_f1_gates.append(f1)
        selected_f2_gates.append(f2)
        if f1 in F_multi:
            F_multi.remove(f1)
        elif f1 in F_single:
            F_single.remove(f1)
        if f2 in F_multi:
            F_multi.remove(f2)
        elif f2 in F_single:
            F_single.remove(f2)
        if S_sel=="s1":
            myDict[g1] = [f1,f2,counter]
            myDict[g2] = [f2,f1,counter-1]
            counter=counter-2
            selected_g1_gates.append(g1)
            selected_g2_gates.append(g2)
            selected_g.append(g1)
            selected_g.append(g2)
            f_link_test_neg.write(str(c.nodes[f2]['count'])+" "+str(c.nodes[g1]['count'])+"\n")
            f_link_test_neg.write(str(c.nodes[f1]['count'])+" "+str(c.nodes[g2]['count'])+"\n")
            f_link_test.write(str(c.nodes[f1]['count'])+" "+str(c.nodes[g1]['count'])+"\n")
            f_link_test.write(str(c.nodes[f2]['count'])+" "+str(c.nodes[g2]['count'])+"\n")
        else:
            if S_sel=="s4":
                selected_g1_gates.append(g1)
                selected_g2_gates.append(g2)
                selected_g.append(g1)
                selected_g.append(g2)
                myDict[g1] = [f1,f2,counter]

                myDict[g2] = [f2,f1,counter]

                f_link_test_neg.write(str(c.nodes[f2]['count'])+" "+str(c.nodes[g1]['count'])+"\n")
                f_link_test_neg.write(str(c.nodes[f1]['count'])+" "+str(c.nodes[g2]['count'])+"\n")
                f_link_test.write(str(c.nodes[f1]['count'])+" "+str(c.nodes[g1]['count'])+"\n")
                f_link_test.write(str(c.nodes[f2]['count'])+" "+str(c.nodes[g2]['count'])+"\n")
            else:

                f_link_test_neg.write(str(c.nodes[f2]['count'])+" "+str(c.nodes[g1]['count'])+"\n")
                f_link_test.write(str(c.nodes[f1]['count'])+" "+str(c.nodes[g1]['count'])+"\n")
                selected_g1_gates.append(g1)
                selected_g.append(g1)
                myDict[g1] = [f1,f2,counter]
            counter=counter-1
    if len(list(nx.simple_cycles(new_c))) !=0:
        sys.exit("There is a loop in the circuit!")
    locked_file = open(location+"/"+benchmark+"_K"+str(key_size)+".bench","w")
    i=0
    locked_file.write("#key=")
    while i<key_size:
        locked_file.write(str(int(K_list[i])))
        i=i+1
    locked_file.write("\n")
    i=0
    while i<key_size:
        locked_file.write("INPUT(keyinput"+str(i)+")\n")
        i=i+1
    file1 = open("./Benchmarks/"+benchmark+".bench", 'r')
    Lines = file1.readlines()
    count = 0
    detected=0
    for line in Lines:
        count += 1
        line=line.strip()
        if any(ext+" =" in line for ext in selected_g):
            detected=detected+1
            regex= "(\S+)\s*=\s*(NOT|BUF|OR|XOR|AND|NAND|XNOR|NOR)\((.+?)\)\s*"
            for output, function, net_str in re.findall(regex,line,flags=re.I |re.DOTALL):
                if output in myDict.keys():
                    my_f1=myDict[output][0]
                    my_f2=myDict[output][1]
                    my_key=myDict[output][2]
                    line=line.replace(my_f1+",", output+"_from_mux,")
                    line=line.replace(my_f1+")", output+"_from_mux)")
                    locked_file.write(line+"\n")
                    if K_list[my_key] == 0:
                        locked_file.write(output+"_from_mux = MUX(keyinput"+str(my_key)+", "+my_f1+", "+my_f2+")\n")
                    else:

                        locked_file.write(output+"_from_mux = MUX(keyinput"+str(my_key)+", "+my_f2+", "+my_f1+")\n")
                else:
                    locked_file.write(line+"\n")
        else:
            locked_file.write(line+"\n")
    locked_file.close()
    f_feat.close()
    f_cell.close()
    f_count.close()
    f_link_test.close()
    f_link_test_neg.close()
    f_link_train.close()
    file1.close()

    with open(location+"/links_test.txt") as f_a, open(location+"/links_train_temp.txt") as f_b:
        a_lines = set(f_a.read().splitlines())
        b_lines = set(f_b.read().splitlines())
        for line in b_lines:
            if line not in a_lines:
                f_link_train_f.write(line+"\n")
    f_link_train_f.close()
    os.remove(location+"/links_train_temp.txt")





