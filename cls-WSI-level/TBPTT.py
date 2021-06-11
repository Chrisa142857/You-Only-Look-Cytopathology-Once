import torch


class TBPTT():
    def __init__(self, one_step_module, loss_module, k1=500, k2=999):
        self.one_step_module = one_step_module
        self.loss_module = loss_module
        self.k1 = k1
        self.k2 = k2
        self.retain_graph = k1 < k2
        # You can also remove all the optimizer code here, and the
        # train function will just accumulate all the gradients in
        # one_step_module parameters

    def train(self, input_sequence, targets, init_state):
        running_loss = []
        hidden_states = [(None, init_state[0])]
        cell_states = [(None, init_state[1])]
        for j in range(input_sequence.shape[1]):
            inp = input_sequence[:, j:j+1, :]
            hidden_state = hidden_states[-1][1].detach()
            hidden_state.requires_grad = True
            cell_state = cell_states[-1][1].detach()
            cell_state.requires_grad = True
            output, new_state = self.one_step_module(inp, (hidden_state, cell_state))
            hidden_states.append((hidden_state, new_state[0]))
            cell_states.append((cell_state, new_state[1]))

            while len(hidden_states) > self.k2:
                # Delete stuff that is too old
                del hidden_states[0]
                del cell_states[0]

            if (j+1) % self.k1 == 0:
                loss = self.loss_module(output, targets.float().cuda())

                # backprop last module (keep graph only if they ever overlap)
                loss.backward(retain_graph=self.retain_graph)
                for i in range(self.k2-1):
                    # if we get all the way back to the "init_state", stop
                    if hidden_states[-i-2][0] is None:
                        break
                    hidden_curr_grad = hidden_states[-i-1][0].grad
                    hidden_states[-i-2][1].backward(hidden_curr_grad, retain_graph=self.retain_graph)
                    cell_curr_grad = cell_states[-i-1][0].grad
                    cell_states[-i-2][1].backward(cell_curr_grad, retain_graph=self.retain_graph)
                running_loss += [loss.item()]
        return torch.FloatTensor([sum(running_loss)/len(running_loss)])