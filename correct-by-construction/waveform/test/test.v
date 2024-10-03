`timescale 1 ps/1 ps
`define OK 12
`define INCORRECT 13

module stimulus_gen (
        input clk,
        output logic a,b,c,d,
        output reg[511:0] wavedrom_title,
        output reg wavedrom_enable
);


// Add two ports to module stimulus_gen:
//    output [511:0] wavedrom_title
//    output reg wavedrom_enable

        task wavedrom_start(input[511:0] title = "");
        endtask

        task wavedrom_stop;
                #1;
        endtask



        initial begin
                {a,b,c,d} <= 0;
                @(negedge clk) wavedrom_start("Unknown circuit");
                        @(posedge clk) {a,b,c,d} <= 0;
                        repeat(15) @(posedge clk, negedge clk) {a,b,c,d} <= {a,b,c,d} + 1;
                wavedrom_stop();
                $finish;
        end

endmodule

module tb();

        wire[511:0] wavedrom_title;
        wire wavedrom_enable;
        int wavedrom_hide_after_time;

        reg clk=0;
        initial forever
                #5 clk = ~clk;

        logic a;
        logic b;
        logic c;
        logic d;
        logic q;

        initial begin 
                $dumpfile("wave.vcd");
                $dumpvars(1, stim1.clk, a,b,c,d,q );
        end


        wire tb_match;          // Verification

        stimulus_gen stim1 (
                .clk,
                .* ,
                .a,
                .b,
                .c,
                .d );
        top_module good1 (
                .a,
                .b,
                .c,
                .d,
                .q(q) );

        bit strobe = 0;
        task wait_for_end_of_timestep;
                repeat(5) begin
                        strobe <= !strobe;  // Try to delay until the very end of the time step.
                        @(strobe);
                end
        endtask

endmodule
