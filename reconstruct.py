import re
import os
import pandas as pd
import numpy as np

"""Script for gap reconstruction in ancestral DNA sequences
"""
__author__ = 'Matej Berezny (xberez03)'
__email__ = "xberez03@stud.fit.vutbr.cz"


class Seqence:
    """Class representation of DNA sequence.
    """
    def __init__(self, name, fasta, mask=None):
        """Constructor for class respresentation of DNA sequence.

        Args:
            name str: name of the sequence
            fasta str: DNA sequence in fasta format
            mask (np.ndarray, optional): masked fasta sequence (-1 for gaps,
                                         1 for aminoacids). Defaults to None.
        """
        self.name = name
        self.fasta = fasta
        self.mask = mask

    def compute_mask(self):
        """Computes the mask from DNA sequence.
        """
        self.mask = np.array(list(self.fasta), dtype=str)
        self.mask = np.where(self.mask == "-", -1, 1)

    def __repr__(self):
        return self.name


class Node:
    """Class representation of one node in phylogenetic tree
    """
    def __init__(self):
        """Node constructor, initializes the instance variables to their default values.
        """
        self.children = []
        self.seq = None
        self.length = 0
        self.leaves = {}
        self.value = None
        
    def isleaf(self):
        """Checks whether node if leaf

        Returns:
            Bool: True if node is leaf node, False otherwise
        """
        return self.value is None

    def save(self, outdir):
        """Saves the ancestral node as .fas file

        Args:
            outdir (str): where to save output sequences
        """
        filename = f"node_{self.seq}.fas"
        with open(os.path.join(outdir, filename), "w") as file:
            file.write(f">{self.seq}\n{self.seq.fasta}")

    def compute_sequence(self):
        """Calculates the ancestral sequence including gaps.
        """
        fasta = ""
        gap_chance = np.zeros(len(self.children[0].seq.fasta))
        # Gather all leaves under current node
        self.leaves = self.gather_leaves(self, 0, 1)
        
        # Calculate the possibility of gap for every amino-acid in sequence
        for leaf, distance in self.leaves.items():
            gap_chance += leaf.seq.mask*distance
        # Construct the ancestral DNA sequence with gap possibilities
        for i in range(0, len(self.children[0].seq.fasta)):
            if gap_chance[i] <= 0:
                fasta+="-"
            else:
                fasta+=self.value.iloc[i].idxmax(axis=1)
        self.seq = Seqence(self.seq, fasta)


    def gather_leaves(self, root, distance, depth):
        """Gathers all leaves under given root node.
        Upon calculation of new upper node, its leaf nodes & 
        distance to them are saved. So there is no need to traverse
        whole tree in every new calculation, algorithm has to just 
        look on node's direct descendants, combine their leaves and 
        update the distance.

        Args:
            root (Node): beginning of the calculation
            distance (int): distance from original root node
            depth (int): depth of the calculation (level of recursion)

        Returns:
            dict: dict containing all leaves and their respective distance from root node
        """
        if root.isleaf():
            root.leaves[root] = distance
        else:
            # Depth was reached, add distance and terminate the calculation
            if depth == 0:
                for leaf in root.leaves:
                    root.leaves[leaf] += distance
            # Continue with recursion
            else:
                depth-=1
                root.leaves.update(self.gather_leaves(root.children[0],
                                                      distance+root.children[0].length,
                                                      depth))
                root.leaves.update(self.gather_leaves(root.children[1],
                                                      distance+root.children[1].length,
                                                      depth))
        return root.leaves


class Tree:
    """Lightweight class representation of a phylogenetic tree
    """
    def __init__(self, msa_file, tree_file, ancestrals_file, outdir, verbose=False):
        """Constructor for Tree class

        Args:
            msa_file (str): file with multiple-sequence alignment in fasta format
            tree_file (str): phylogenetic tree in newick format
            ancestrals_file (str): file with posterior probabilities
            oudir (str): where to save output sequences
            verbose (bool): If True, prints the computed sequences to stdout.
                            Defaults to False.
        """
        self.root = None
        self.outdir = outdir
        self.verbose = verbose
        self.__build(tree_file, msa_file, ancestrals_file)


    def traverse(self, root):
        """Traverses the entire phylogenetic tree
        and computes sequences for non-leaf nodes

        Args:
            root (Node): root of the tree
        """
        if not isinstance(root.seq, Seqence):
            self.traverse(root.children[0])
            self.traverse(root.children[1])
            root.compute_sequence()
            root.save(self.outdir)
            if self.verbose:
                print(f"{root.seq} --> {root.seq.fasta}")



    def __build(self, tree_file, msa_file, ancestrals):
        """Constructs the phylogenetic tree from file
        in newick format

        Args:
            msa_file (str): file with multiple-sequence alignment in fasta format
            tree_file (str): phylogenetic tree in newick format
            ancestrals_file (str): file with posterior probabilities
        """
        self.seq_list = {}

        # Process file with multiple-sequence alignment
        with open(msa_file, "r") as file:
            content = file.read()
        for item in content.split(">"):
            splitted = item.split("\n")
            name = splitted[0]
            seq = "".join(splitted[1:])
            self.seq_list[name] = Seqence(name, seq)
        
        # Load the .csv file and convert it to numeric
        self.ancestrals = pd.read_csv(ancestrals)
        for column in self.ancestrals.columns:
            self.ancestrals[column] = pd.to_numeric(self.ancestrals[column], errors='coerce')
        
        # Read tree file
        with open(tree_file, "r") as file:
            content = file.read()
        self.root, _ = self.__parse(content, 0)

        # Prepare folder for output sequences
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)


    def __read_branch_len(self, tstring, index):
        """Character-wise parsing of length of individual branches
        from the input string in newick format.

        Args:
            tstring (str): phylogenetic tree in newick format
            index (int): current index in tstring

        Returns:
            str: name of the node 
        """
        end = index
        while True:
            end += 1
            if end==len(tstring):
                break
            if tstring[end]==',' or tstring[end]==')':
                break
        blen = float( tstring[index+1:end] )
        return blen, end


    def __read_node_name(self, tstring, index):
        """Character-wise parsing of node name from the input
        string in newick format. Name can be an actual DNA sequence
        found in msv_file, or simple node index found in ancestrals
        file.

        Args:
            tstring (str): phylogenetic tree in newick format
            index (int): current index in tstring

        Returns:
            str: name of the node 
        """
        end = index
        while True:
            if end == len(tstring):
                break
            if tstring[end] == ":" or tstring[end] == ";":
                break
            end += 1
        name = tstring[index:end]
        return name, end


    def __read_leaf(self, tstring, index):
        """Character-wise parsing of leaf node from input
        string in newick format. The function returns leaf node
        with DNA sequence along with DNA sequence assigned to it.

        Args:
            tstring (str): phylogenetic tree in newick format
            index (int): current index in tstring

        Returns:
            Node: leaf node
        """
        end = tstring.find(":", index)
        node = Node()
        node.seq = self.seq_list[tstring[index:end]]
        node.length, index = self.__read_branch_len(tstring, end)
        node.seq.compute_mask()
        return node, index


    def __parse(self, tstring, index):
        """Character-wise recursive parsing of input tree
        string in newick format. The function constructs whole
        phylogenetic tree structure and returns pointer to the root
        node.

        Args:
            tstring (str): phylogenetic tree in newick format
            index (int): current index in tstring

        Returns:
            Node: root node of the built tree
        """
        index += 1
        node = Node()
        while True:
            if tstring[index] == '(':
                child, index = self.__parse(tstring, index)
                node.children.append(child)
            elif tstring[index] == ",":
                index+=1
            elif tstring[index] == ")":
                index+=1
                if index<len(tstring):
                    if re.match(r"^[A-Za-z0-9]", tstring[index]):
                        node.seq, index = self.__read_node_name(tstring, index)
                        # Add node-specific posterior probabilities
                        node.value = self.ancestrals[self.ancestrals.node == int(node.seq)].set_index('position')
                        node.value = node.value.drop(['node'], axis=1)
                    if index == len(tstring):
                        break
                    if tstring[index] == ':':
                        node.length, index = self.__read_branch_len(tstring, index)
                break
            else:
                child, index = self.__read_leaf(tstring, index)
                node.children.append(child)
        return node, index


if __name__ == "__main__":
    tree = Tree(msa_file="msa.fasta",
                tree_file="tree.tre",
                ancestrals_file="ancestrals.csv",
                outdir="output_sequences")
    tree.traverse(tree.root)