#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import enum


class PIPELINE_TYPE(enum.Enum):
    TXT2IMG = enum.auto()
    IMG2IMG = enum.auto()
    IMG2VID = enum.auto()
    INPAINT = enum.auto()
    CONTROLNET = enum.auto()
    XL_CONTROLNET = enum.auto()
    XL_BASE = enum.auto()
    XL_REFINER = enum.auto()
    CASCADE_PRIOR = enum.auto()
    CASCADE_DECODER = enum.auto()
    VIDEO2WORLD = enum.auto()

    def is_txt2img(self):
        return self in (self.TXT2IMG, self.CONTROLNET)

    def is_img2img(self):
        return self == self.IMG2IMG

    def is_img2vid(self):
        return self == self.IMG2VID

    def is_inpaint(self):
        return self == self.INPAINT

    def is_controlnet(self):
        return self in (self.CONTROLNET, self.XL_CONTROLNET)

    def is_sd_xl_base(self):
        return self in (self.XL_BASE, self.XL_CONTROLNET)

    def is_sd_xl_refiner(self):
        return self == self.XL_REFINER

    def is_sd_xl(self):
        return self.is_sd_xl_base() or self.is_sd_xl_refiner()

    def is_cascade_prior(self):
        return self == self.CASCADE_PRIOR

    def is_cascade_decoder(self):
        return self == self.CASCADE_DECODER

    def is_cascade(self):
        return self.is_cascade_prior() or self.is_cascade_decoder()

    def is_video2world(self):
        return self == self.VIDEO2WORLD
