from cryptorch.utils import _export_module, get_graph_cost

class KnobTuner:
    def __init__(self):
        pass
    def reset(self):
        raise NotImplementedError()
    def generate_next_candidate(self, _match, knobs, last_attempt_successful):
        raise NotImplementedError()

class LinearGreedyKnobTuner(KnobTuner):
    def __init__(self):
        super().__init__()
        self.i = 0
        self.j = 0
        self.cur_state = []
        self.prev_state = []

    def reset(self):
        self.i = 0 # operator index
        self.j = 0 # knob index within the operator
        self.cur_state = []
        self.prev_state = []

    def generate_next_candidate(self, _match, last_attempt_successful):
        # Simple greedy linear search.
        # We linearly go through the operators from the beginning.
        # For each operator, we go through all the knobs and try to greedily reduce each knob.
        if not last_attempt_successful:
            # Revert to the previous setup
            assert(len(self.prev_state) > 0)
            self.cur_state = self.prev_state + []
            # Move to the next knob.
            if not self.increment():
                # If no more knobs to tune, finish.
                return []
        
        # Try to decrement the i-th operator's j-th knob.
        self.prev_state = self.cur_state + []
        while True:
            p = _match[self.i][0]
            knob_to_tune = self.cur_state[self.i]
            new_knob = p.decrement_knob(knob_to_tune, self.j)
            if new_knob is not None:
                # If the selected knob is decrementable,
                # decrement and return.
                self.cur_state[self.i] = new_knob
                return self.cur_state

            # Else, move to the next knob and try again.
            if not self.increment():
                # If no more knobs to tune, finish.
                return []

    def increment(self):
        if self.j == len(self.cur_state[self.i]) - 1:
            # If there is no other knobs in this operator.
            if self.i == len(self.cur_state) - 1:
                # If this is the last operator.
                return False
            else:
                # Else, move to the next operator.
                self.i += 1
                self.j = 0
        else:
            # Else, move on to the next knob.
            self.j += 1
        return True

class HillClimbingKnobTuner(KnobTuner):
    def __init__(self):
        super().__init__()
        # self.intp = intp # Interpreter needed to calculate the cost
        self.prev_i = 0 # Last modified operator index
        self.prev_knobs = None # Last modified operator's knob values before trying to optimized
        self.cur_state = []
        self.costs = dict() # (i, k): cost, Cost of using knobs k for the i-th op.

    def reset(self):
        self.prev_i = 0
        self.prev_knobs = None
        self.cur_state = []
        self.costs = dict()

    def generate_next_candidate(self, _match, last_attempt_successful):
        if not last_attempt_successful:
            # Revert to the previous setup
            assert(len(self.prev_knobs) is not None)
            # Never choose this (i, k) pair again.
            # This is not the most efficient implementation, because anything more aggressive than this approx should also never be chosen, which this does not check. For example, if (0, 1, 6) is not valid, probably (0, 0, 6) or (0, 1, 4) is also not valid. However, currently the inefficiency isn't too high.
            bad_knobs = self.cur_state[self.prev_i]
            # Anything more aggressive than this approx should also never be chosen.
            # For example, if (0, 1, 6) is not valid,
            # probably (0, 0, 6), (0, 1, 4), (0, 1, 2) are also not valid.
            self.costs[(self.prev_i, bad_knobs)] = 999999999999999999
            p = _match[self.prev_i][0]
            for j in range(len(bad_knobs)):
                also_bad_knobs = bad_knobs
                while True:
                    also_bad_knobs = p.decrement_knob(also_bad_knobs, j)
                    if also_bad_knobs is None:
                        break
                    self.costs[(self.prev_i, also_bad_knobs)] = 999999999999999999999999999999999

            self.cur_state[self.prev_i] = self.prev_knobs

        # Choose the direction with the steepest cost benefit.
        best_benefit = -1
        best_state = None

        for i, (p, m, inputs) in enumerate(_match):
            for j in range(len(self.cur_state[i])):
                new_state = self.cur_state + []
                new_knobs = p.decrement_knob(self.cur_state[i], j)
                if new_knobs is None:
                    continue
                cur_cost = self.get_cost(i, self.cur_state[i], p, inputs)
                new_cost = self.get_cost(i, new_knobs, p, inputs)
                benefit = cur_cost - new_cost
                print(f"Reducing {i}-th op from {self.cur_state[i]} to {new_knobs} benefit {benefit})")
                if benefit > best_benefit:
                    print("Saving as best")
                    best_benefit = benefit
                    best_state = self.cur_state + []
                    best_state[i] = new_knobs
                    self.prev_i = i

        if best_state is None:
            return []

        self.prev_knobs = self.cur_state[self.prev_i]
        self.cur_state = best_state + []
        return self.cur_state

    def get_cost(self, i, knobs, p, inputs):
        #print(i, knobs, p, inputs)
        if (i, knobs) in self.costs:
            cost = self.costs[(i, knobs)]
            print(f"Cost cached for {(i, knobs)}, {cost=}")
        else:
            replacement = p.get_replacement(knobs)
            replacement_graph = _export_module(replacement, inputs).graph
            cost = get_graph_cost(replacement_graph)
            print(f"Cost for {(i, knobs)}, {cost=}")
            self.costs[(i, knobs)] = cost
        return cost

class BinarySearchGreedyKnobTuner(KnobTuner):
    def __init__(self):
        super().__init__()
        self.i = 0
        self.j = 0
        self.cur_state = []
        self.prev_state = []

    def reset(self):
        self.i = 0 # operator index
        self.j = 0 # knob index within the operator
        self.cur_state = []
        self.prev_state = []
        self.possible_knob_positions = []

    def generate_next_candidate(self, _match, last_attempt_successful):
        if len(self.possible_knob_positions) == 0:
            self.setup_binary_search(_match)
        
        # update search space
        if self.test_index >= 0:
            if last_attempt_successful:
                self.known_good = self.test_index
            else:
                self.known_bad = self.test_index
        
        # check if search is complete
        if self.known_good == self.known_bad + 1:
            # this knob is done
            current_knobs = list(self.cur_state[self.i])
            current_knobs[self.j] = self.possible_knob_positions[self.known_good]
            self.cur_state[self.i] = current_knobs
            # go to next step
            if not self.increment():
                return []
            self.setup_binary_search(_match)
        
        # start next test
        self.test_index = (self.known_bad + self.known_good) // 2
        current_knobs = list(self.cur_state[self.i])
        current_knobs[self.j] = self.possible_knob_positions[self.test_index]
        
        new_knobs = [k for k in self.cur_state]
        new_knobs[self.i] = current_knobs
        
        return new_knobs

    def increment(self):
        if self.j == len(self.cur_state[self.i]) - 1:
            # If there is no other knobs in this operator.
            if self.i == len(self.cur_state) - 1:
                # If this is the last operator.
                return False
            else:
                # Else, move to the next operator.
                self.i += 1
                self.j = 0
        else:
            # Else, move on to the next knob.
            self.j += 1
            
        # setup binary search parameters
        
        return True
    
    def setup_binary_search(self, _matches):
        self.possible_knob_positions = _matches[self.i][0].get_possible_knob_positions()[self.j]
        self.known_bad = -1
        self.known_good = len(self.possible_knob_positions) - 1
        self.test_index = -1