import sys
# import .genotypes
from graphviz import Digraph

def plot(genotype, filename, args, task=None):
    
    if genotype == None:
        return 

    multiplier = args.multiplier
    num_input_nodes = args.num_input_nodes
    num_keep_edges = args.num_keep_edges
    
    node_steps = args.node_steps
    node_multiplier = args.node_multiplier
    
    g = Digraph(
            format='pdf',
            edge_attr=dict(fontsize='20', fontname="times", penwidth='1.5'),
            node_attr=dict(style='rounded, filled', shape='rect', align='center', 
                           fontsize='20', height='0.5', width='0.5', penwidth='2', 
                           fontname="helvetica"),
            engine='dot')
    # g.attr(style='rounded, filled', color='red')
    # g.body.extend(['rankdir=LR'])
    g.attr(rankdir='LR')

    input_nodes = []
    input_nodes_A = []
    input_nodes_B = []
    
    nc = Digraph(node_attr={'shape': 'box'})
    nc.attr(rankdir='BT')
    nc.attr(rank='same')
    # nc.attr(style='rounded, filled', color='red')
    #                fontsize='20', align='center')
    # with nc.subgraph(name='cluster_input_features', node_attr={'shape': 'box'}) as c:
    #     c.attr(style='rounded, filled', color='red', 
    #                fontsize='20', align='center')
    #     c.attr(rankdir='BT')
    #     c.attr(rank='same')
    #     c.attr(constraint='false')
    assert len(genotype.edges) % num_keep_edges == 0
    steps = len(genotype.edges) // num_keep_edges
    
    with g.subgraph() as nothing:
        c = nc
        
        input_nodes_A = []
        input_nodes_B = []
        
        with c.subgraph(name='cluster_video_features', node_attr={'shape': 'box'}) as ca:
            ca.attr(style='rounded, filled', color='lightgrey', 
                           fontsize='20', align='center')

            input_nodes_A = ["Video_1", "Video_2", "Video_3", "Video_4"]
            if task == 'mmimdb':
                input_nodes_A = ["Image_1", "Image_2", "Image_3", "Image_4"]
            
            if task == 'nvgesture' or task == 'ego':
                input_nodes_A = ["RGB_1", "RGB_2", "RGB_3", "RGB_4"]
                
            for input_node in input_nodes_A:
                ca.node(input_node, fillcolor='lightskyblue1')
            
            for i in range(len(input_nodes_A)-1):
                c.edge(input_nodes_A[i], input_nodes_A[i+1], label=None)
            
            
    
        with c.subgraph(name='cluster_skeleton_features', node_attr={'shape': 'box'}) as cb:
            cb.attr(style='rounded, filled', color='lightgrey', 
                           fontsize='20', align='center')

            input_nodes_B = ["Skeleton_1", "Skeleton_2", "Skeleton_3", "Skeleton_4"]
            if task == 'mmimdb':
                input_nodes_B = ["Text_1", "Text_2"]
            
            if task == 'nvgesture' or task == 'ego':
                input_nodes_B = ["Depth_1", "Depth_2", "Depth_3", "Depth_4"]

            for input_node in input_nodes_B:
                cb.node(input_node, fillcolor='darkolivegreen1')
            
            for i in range(len(input_nodes_B)-1):
                c.edge(input_nodes_B[i], input_nodes_B[i+1], label=None)
                
                # for j in range(steps):        
                #     node_x_name = "X_C{}".format(j)
                #     node_y_name = "Y_C{}".format(j)
                #     # g.edge(input_nodes_B[i], node_y_name, style='invis')
                #     g.edge(input_nodes_B[i], node_x_name, style='invis')
            
        c.edge(input_nodes_A[-1], input_nodes_B[0], style='invis')
        
    g.subgraph(nc)
    
    input_nodes = input_nodes_A + input_nodes_B
    assert len(input_nodes) == num_input_nodes

    node_names = [] 
    node_names += input_nodes

    for i in range(steps):
        # step_op = genotype.steps[i][0]
        # step_node_name = "{}_{}".format(i, step_op)
        node_z_name = "Z_C{}".format(i+1)
        node_names.append(node_z_name)
    
    # for i in range(steps-1):        
    #     node_x_name = "X_C{}".format(i+2)
    #     node_y_name = "Y_C{}".format(i+1)
    #     g.edge(node_x_name, node_y_name, style='invis')
    #     g.edge(node_y_name, node_x_name, style='invis')

    
    # for i in genotype.concat[0:-1]:
    #     # print(i)
    #     g.edge(node_names[i+1], node_names[i], style='invis')
        
    for i in range(steps):
        # step_op = genotype.steps[i][0]
        # step_node_name = "{}_{}".format(i, step_op)
        step_node_name = "cluster_step_{}".format(i)
        step_gene = genotype.steps[i]
        
        node_x_name = "X_C{}".format(i+1)
        node_y_name = "Y_C{}".format(i+1)
        node_z_name = "Z_C{}".format(i+1)

        with g.subgraph(name=step_node_name, node_attr={'shape': 'box'}) as c:
            c.attr(style='rounded, filled', color='tan1', 
                   fontsize='20', align='center')
            c.node_attr.update(style='rounded, filled')
            
            inner_node_names = [node_x_name, node_y_name]
            for j in range(node_steps):    
                # print(i, j)
                # print(step_gene)
                inner_step_name = "C{}_S{}\n{}".format(i+1, j+1, step_gene.inner_steps[j]) 
                inner_node_names.append(inner_step_name)
            
            with c.subgraph() as ic:
                # ic.attr(rankdir='BT')
                # ic.attr(rank='same')
                for inner_node_name in inner_node_names:
                    if inner_node_name != node_x_name and inner_node_name != node_y_name:
                        ic.node(inner_node_name, fillcolor='khaki1')
            
            c.node(node_x_name, fillcolor='maroon2')
            c.node(node_y_name, fillcolor='green3')
            c.node(node_z_name, fillcolor='purple')
            
            # c.edge(input_nodes_B[-1], node_x_name, style='invis')
            # c.edge(input_nodes_B[-1], node_y_name, style='invis')
            # for in_A in input_nodes_A:
            #     c.edge(in_A, node_x_name, style='invis')
            #     c.edge(in_A, node_y_name, style='invis')
            
            # for in_B in input_nodes_B:
            #     c.edge(in_B, node_x_name, style='invis')
            #     c.edge(in_B, node_y_name, style='invis')

            # print(inner_node_names)
            for j in range(node_steps):
                x = step_gene.inner_edges[2*j][1]
                x_op = step_gene.inner_edges[2*j][0]
                y = step_gene.inner_edges[2*j+1][1]
                y_op = step_gene.inner_edges[2*j+1][0]
                # print(j, x, x_op, y, y_op)
                # c.edge(inner_node_names[x], inner_node_names[2+j], label=x_op)
                # c.edge(inner_node_names[y], inner_node_names[2+j], label=y_op)
                c.edge(inner_node_names[x], inner_node_names[2+j], label=None)
                c.edge(inner_node_names[y], inner_node_names[2+j], label=None)
                
                # c.edge(inner_node_names[2+j], inner_node_names[x], label=None)
                # c.edge(inner_node_names[2+j], inner_node_names[y], label=None)
            
            for j in range(args.node_multiplier):
                # c.edge(inner_node_names[-(j+1)], node_z_name, label='skip')
                c.edge(inner_node_names[-(j+1)], node_z_name, label=None)
            # skip connection
            # c.edge(node_x_name, node_z_name, label=None)
        
        edge_x_op = genotype.edges[2*i][0]
        edge_x_from = node_names[genotype.edges[2*i][1]]
        edge_x_to = node_x_name
        # g.edge(edge_x_from, edge_x_to, label=edge_x_op)
        g.edge(edge_x_from, edge_x_to, label=None, color="blue")

        edge_y_op = genotype.edges[2*i+1][0]
        edge_y_from = node_names[genotype.edges[2*i+1][1]]
        edge_y_to = node_y_name
        # g.edge(edge_y_from, edge_y_to, label=edge_y_op)
        g.edge(edge_y_from, edge_y_to, label=None, color="blue")
    
    g.node("Reduction\nOutput", fillcolor='grey91')

    for i in genotype.concat:
        g.edge(node_names[i], "Reduction\nOutput", color="blue")

    g.render(filename, view=False)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("usage:\n python {} ARCH_NAME".format(sys.argv[0]))
        sys.exit(1)

    genotype_name = sys.argv[1]
    try:
        genotype = eval('genotypes.{}'.format(genotype_name))
    except AttributeError:
        print("{} is not specified in genotypes.py".format(genotype_name)) 
        sys.exit(1)

    plot(genotype.normal, "normal")
'''
def plot(genotype, filename, args, task=None):
    if genotype == None:
        return 

    multiplier = args.multiplier
    num_input_nodes = args.num_input_nodes
    num_keep_edges = args.num_keep_edges
    
    node_steps = args.node_steps
    node_multiplier = args.node_multiplier
    
    g = Digraph(
            format='pdf',
            edge_attr=dict(fontsize='20', fontname="times"),
            node_attr=dict(style='filled', shape='rect', align='center', 
                           fontsize='20', height='0.5', width='0.5', penwidth='2', 
                           fontname="times"),
            engine='dot')
    g.body.extend(['rankdir=LR'])

    input_nodes = []
    input_nodes_A = []
    input_nodes_B = []
    
    with g.subgraph(name='cluster_input_features', node_attr={'shape': 'box'}) as c:
        c.attr(style='rounded, filled', color='white', 
                   fontsize='20', align='center')
        
        with c.subgraph(name='cluster_video_features', node_attr={'shape': 'box'}) as ca:
            ca.attr(style='rounded, filled', color='lightgrey', 
                           fontsize='20', align='center')
            input_nodes_A = ["video_0", "video_1", "video_2", "video_3"]
            if task == 'mmimdb':
                input_nodes_A = ["image_0", "image_1", "image_2", "image_3"]
            
            if task == 'nvgesture' or task == 'ego':
                input_nodes_A = ["rgb_0", "rgb_1", "rgb_2", "rgb_3"]
                
            for input_node in input_nodes_A:
                ca.node(input_node, fillcolor='darkseagreen2')
    
        with c.subgraph(name='cluster_skeleton_features', node_attr={'shape': 'box'}) as cb:
            cb.attr(style='rounded, filled', color='lightgrey', 
                           fontsize='20', align='center')

            input_nodes_B = ["skeleton_0", "skeleton_1", "skeleton_2", "skeleton_3"]
            if task == 'mmimdb':
                input_nodes_B = ["text_0", "text_1"]
            
            if task == 'nvgesture' or task == 'ego':
                input_nodes_B = ["depth_0", "depth_1", "depth_2", "depth_3"]

            for input_node in input_nodes_B:
                cb.node(input_node, fillcolor='darkseagreen2')
    
    input_nodes = input_nodes_A + input_nodes_B
    assert len(input_nodes) == num_input_nodes

    assert len(genotype.edges) % num_keep_edges == 0
    steps = len(genotype.edges) // num_keep_edges

    node_names = [] 
    node_names += input_nodes

    
    for i in range(steps):
        # step_op = genotype.steps[i][0]
        # step_node_name = "{}_{}".format(i, step_op)
        step_node_name = "cluster_step_{}".format(i)
        step_gene = genotype.steps[i]
        
        node_x_name = "step_{}_x".format(i)
        node_y_name = "step_{}_y".format(i)

        node_z_name = ''
        if args.node_multiplier == 1:
            node_z_name = "step_{}_z".format(i)
        else:
            node_z_name = "step_{}_z\ncat_conv_relu".format(i)
        
        node_names.append(node_z_name)
        
        with g.subgraph(name=step_node_name, node_attr={'shape': 'box'}) as c:
            c.attr(style='rounded, filled', color='lightgrey', 
                   fontsize='20', align='center')
            c.node_attr.update(style='rounded, filled')
            
            inner_node_names = [node_x_name, node_y_name]
            for j in range(node_steps):    
                inner_step_name = "step_{}_{}\n{}".format(i, j, step_gene.inner_steps[j]) 
                inner_node_names.append(inner_step_name)
                
            for inner_node_name in inner_node_names:
                c.node(inner_node_name, fillcolor='orange')
            
            c.node(node_x_name, fillcolor='lightblue')
            c.node(node_y_name, fillcolor='lightblue')
            c.node(node_z_name, fillcolor='lightblue')
            # print(inner_node_names)
            for j in range(node_steps):
                x = step_gene.inner_edges[2*j][1]
                x_op = step_gene.inner_edges[2*j][0]
                y = step_gene.inner_edges[2*j+1][1]
                y_op = step_gene.inner_edges[2*j+1][0]
                # print(j, x, x_op, y, y_op)
                # c.edge(inner_node_names[x], inner_node_names[2+j], label=x_op)
                # c.edge(inner_node_names[y], inner_node_names[2+j], label=y_op)
                c.edge(inner_node_names[x], inner_node_names[2+j], label=None)
                c.edge(inner_node_names[y], inner_node_names[2+j], label=None)
            for j in range(args.node_multiplier):
                # c.edge(inner_node_names[-(j+1)], node_z_name, label='skip')
                c.edge(inner_node_names[-(j+1)], node_z_name, label=None)

            c.edge(node_x_name, node_z_name, label=None)
        
        edge_x_op = genotype.edges[2*i][0]
        edge_x_from = node_names[genotype.edges[2*i][1]]
        edge_x_to = node_x_name
        # g.edge(edge_x_from, edge_x_to, label=edge_x_op)
        g.edge(edge_x_from, edge_x_to, label=None)

        edge_y_op = genotype.edges[2*i+1][0]
        edge_y_from = node_names[genotype.edges[2*i+1][1]]
        edge_y_to = node_y_name
        # g.edge(edge_y_from, edge_y_to, label=edge_y_op)
        g.edge(edge_y_from, edge_y_to, label=None)
    
    g.node("out\ncat_conv_relu", fillcolor='palegoldenrod')

    for i in genotype.concat:
        g.edge(node_names[i], "out\ncat_conv_relu", fillcolor="gray")

    g.render(filename, view=False)
'''