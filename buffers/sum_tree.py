# From: https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/tree.py

# The ‘sum-tree’ data structure used here is very similar in spirit to the array representation
# of a binary heap. However, instead of the usual heap property, the value of a parent node is
# the sum of its children. Leaf nodes store the transition priorities and the internal nodes are
# intermediate sums, with the parent node containing the sum over all priorities, p_total. This
# provides a efficient way of calculating the cumulative sum of priorities, allowing O(log N) updates
# and sampling. (Appendix B.2.1, Proportional prioritization)


class SumTree:
    def __init__(self, memory_size: int):
        self.nodes = [0] * (2 * memory_size - 1)
        self.data = [None] * memory_size

        # memory_size = size
        # memory_pointer = count
        # memory_num_experiences = real_size

        self.memory_size = memory_size
        self.memory_pointer = 0
        self.memory_num_experiences = 0

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.memory_size - 1  # child index in tree array
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.memory_pointer] = data
        self.update(self.memory_pointer, value)

        # Increment pointers & capacity
        self.memory_pointer = (self.memory_pointer + 1) % self.memory_size
        self.memory_num_experiences = min(self.memory_size, self.memory_num_experiences + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2 * idx + 1, 2 * idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.memory_size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]

    def __repr__(self):
        return f"SumTree(nodes={self.nodes.__repr__()}, data={self.data.__repr__()})"
