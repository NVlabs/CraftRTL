# Copyright (c) 2024, NVIDIA Corporation. All rights reserved.
#
# This work is made available
# under the Nvidia Source Code License (1-way Commercial).

from __future__ import print_function

from typing import Optional, Callable, Dict
import ast
import contextlib
import faulthandler
import io
import os
import multiprocessing
import platform
import signal
import tempfile

import subprocess
import re
from threading import Timer


import sys
from pprint import PrettyPrinter
import math
import vcdvcd
from vcdvcd import VCDVCD, binary_string_to_hex

from verilog_eval.execution import *

class MyStreamParserCallbacks(vcdvcd.PrintDumpsStreamParserCallbacks):
    def enddefinitions(
        self,
        vcd,
        signals,
        cur_sig_vals
    ):
        self.sequence = ['clk', 'reset', 'in', 'out']
        if signals:
            self._print_dumps_refs = signals
        else:
            self._print_dumps_refs = sorted(vcd.data[i].references[0] for i in cur_sig_vals.keys())
        for i, ref in enumerate(self._print_dumps_refs, 1):
            #print('{} {}'.format(i, ref))
            if i == 0:
                i = 1
            identifier_code = vcd.references_to_ids[ref]
            size = int(vcd.data[identifier_code].size)
            width = max(((size // 4)), int(math.floor(math.log10(i))) + 1)
            self._references_to_widths[ref] = width
        self.output = []
        line = "// " + "time".ljust(16)
        #for i, ref in enumerate(self._print_dumps_refs):
        for ref in self.sequence:
            ref = ref.split('.')[-1]
            line += f"{ref}".ljust(16)
        self.output.append(line)
        #print(line)
        #print('=' * (sum(self._references_to_widths.values()) + len(self._references_to_widths) + 1))

    def time(
        self,
        vcd,
        time,
        cur_sig_vals
    ):
        if (not self._deltas or vcd.signal_changed):
            line = "// " + f"{time}ns".ljust(16)
            self.signal_values = {}
            for i, ref in enumerate(self._print_dumps_refs):
                identifier_code = vcd.references_to_ids[ref]
                value = cur_sig_vals[identifier_code]
                self.signal_values[ref.split('.')[-1]] = value
            for ref in self.sequence:
                line += f"{binary_string_to_hex(self.signal_values[ref])}".ljust(16)
                #ss.append('{0:>{1}s}'.format(
                #    binary_string_to_hex(value),
                #    self._references_to_widths[ref])
                #)
            self.output.append(line)
            #print(line)

def obtain_waveform(testbench: str, timeout=90) -> Dict:
    """
    Evaluates the functional correctness of a completion by running the test
    suite provided in the problem. 
    :param completion_id: an optional completion ID so we can match
        the results later even if execution finishes asynchronously.
    """

    def unsafe_execute():

        with create_tempdir():

            # These system calls are needed when cleaning up tempdir.
            import os
            import shutil
            rmtree = shutil.rmtree
            rmdir = os.rmdir
            chdir = os.chdir

            reliability_guard()

            verilog_test = testbench

                
                    
            with open("test.sv".format(), 'w') as f:
                f.write(verilog_test)
            
            try:

                with swallow_io():
                    with time_limit(timeout):
                        cmd = "iverilog -Wall -Winfloop -Wno-timescale -g2012 \
                                    -s tb -o test.vvp test.sv; vvp -n test.vvp"
                       
                        """
                        adding timeout options for Popen. something breaks if not using timeout. seems to be working for now.
                        not really sure if its the best/correct way. let me know if anyone has a better solution.
                        https://stackoverflow.com/questions/1191374/using-module-subprocess-with-timeout
                        """
                        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        timer = Timer(timeout, p.kill)
                        try:
                            timer.start()
                            out, err = p.communicate()
                        finally:
                            timer.cancel()
                            
                        try:
                            cbk = MyStreamParserCallbacks()
                            vcd = VCDVCD('wave.vcd', callbacks=cbk, store_tvs=False)
                            output = "\n".join(cbk.output)
                            result.append(output)
                        except:
                            result.append("Waveform not exists.")
                        

            except TimeoutException:
                result.append("timed out")
            except BaseException as e:
                result.append(f"failed: {e}")

            # Needed for cleaning up.
            shutil.rmtree = rmtree
            os.rmdir = rmdir
            os.chdir = chdir
            
    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return result


code = """module top_module (
        input a, 
        input b, 
        input c, 
        input d,
        output q
);

        assign q = (a|b) & (c|d);

endmodule"""
with open('test/test.v', 'r') as f:
    test = f.read()

test += "\n" + code
res = obtain_waveform(test)
#print(res)
