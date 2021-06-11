import torch.nn as nn
import torch

class straightResModule(nn.Module):

    def __init__(self, ind_num=2, ind_id=0, in_channels=1024, out_channels=1024, pad=1, stride=1, bn=True):
        super(straightResModule, self).__init__()
        self.ind_conv_in = int(in_channels / ind_num)
        if ind_id >= ind_num:
            print("ERROR in sRes init")
            exit()
        self.conv33 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=pad, stride=stride, bias=not bn, groups=in_channels)
        # self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=stride, bias=not bn, groups=1)
        self.conv11 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1, bias=not bn, groups=1)
        # self.ind_id = ind_id

    def forward(self, x):
        x = self.conv33(x)
        x = self.conv11(x)
        return x

class sResModuleBlock(nn.Module):

    def __init__(self, block_id=0, ind_num=2, in_channels=1024, out_channels=1024, pad=1, stride=1, bn=True):
        super(sResModuleBlock, self).__init__()
        module_list = nn.ModuleList()
        self.ind_ids = []
        self.ind_conv_in = int(in_channels/ind_num)
        self.ind_conv_out = int(out_channels/ind_num)
        self.stride = stride
        self.pad = pad
        filters = self.ind_conv_in
        for i in range(ind_num):
            # if i == ind_num-1:
            #     filters = self.ind_conv_in
            # else:
            #     filters = self.ind_conv_in
            modules = nn.Sequential()
            self.ind_ids.append(i)
            modules.add_module(
                f"sResModule_{i+block_id*ind_num}",
                straightResModule(
                    ind_num=ind_num,
                    ind_id=i,
                    in_channels=self.ind_conv_in,
                    out_channels=filters,
                    pad=pad,
                    stride=1,
                    bn=bn,
                )
            )
            if bn:
                modules.add_module(
                f"sRM_bn_{i+block_id*ind_num}",
                    nn.BatchNorm2d(filters, momentum=0.9, eps=1e-5)
                )
            module_list.append(modules)
        self.module_list = module_list
        self.final_conv = nn.Sequential()
        # self.final_conv.add_module(
        #     f"sRM_final_conv33_{block_id}",
        #     nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1, stride=stride, bias=not bn, groups=in_channels)
        # )
        # if bn:
        #     self.final_conv.add_module(
        #         f"sRM_final_bn33_{block_id}",
        #         nn.BatchNorm2d(in_channels, momentum=0.9, eps=1e-5)
        #     )
        self.final_conv.add_module(
            f"sRM_final_conv11_{block_id}",
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0, stride=1, bias=not bn, groups=1)
        )
        if bn:
            self.final_conv.add_module(
                f"sRM_final_bn11_{block_id}",
                nn.BatchNorm2d(out_channels, momentum=0.9, eps=1e-5)
            )
        self.final_conv.add_module(
            f"sRM_final_avgpool_{block_id}",
            nn.AvgPool2d(kernel_size=stride, stride=stride)
        )
        # self.max_pool = nn.MaxPool2d(kernel_size=stride, stride=stride)

    def forward(self, x):
        # B, C, H, W = x.shape
        # x = x.view()
        # ind_xs = [x[:, ind_id * self.ind_conv_in:(ind_id + 1) * self.ind_conv_in] for ind_id in self.ind_ids]
        # ind_xs = [module(x[:, ind_id * self.ind_conv_in:(ind_id + 1) * self.ind_conv_in]) for module, ind_id in zip(self.module_list, self.ind_ids)]
        # x = self.avg_pool(x)
        ind_xs = []
        for module, ind_id in zip(self.module_list, self.ind_ids):
            ind_x = module(x[:, ind_id * self.ind_conv_in:(ind_id + 1) * self.ind_conv_in])
            for i in self.ind_ids:
                if i == ind_id: continue
                if self.pad == 1: 
                    ind_x += x[:, i * self.ind_conv_in:(i + 1) * self.ind_conv_in]
                elif self.pad == 0:
                    ind_x += x[:, i * self.ind_conv_in:(i + 1) * self.ind_conv_in, 1:-1, 1:-1]
            ind_xs.append(ind_x)
        x = torch.cat(ind_xs, dim=1)
        # x = torch.cat([ind_x+xi for ind_x, ind_id in zip(ind_xs, self.ind_ids)], dim=1)
        return self.final_conv(x)



