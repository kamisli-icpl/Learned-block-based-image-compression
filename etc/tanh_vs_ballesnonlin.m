

x=-5:0.1:5 ;

cntr = 1 ; figure; hold on;
for a = -1:0.2:1
    
    gkx{cntr} = x + a .* tanh(x) ;
    
    plot(x, gkx{cntr}); 
    cntr = cntr + 1 ;
    
end
hold off;
grid on;



x = [-255:0.1:255]*5 ;

cntr = 1 ; figure; hold on;
for a = -1:0.2:1
    
    gkx{cntr} = x + a .* tanh(x/256)*256 ;
    
    plot(x, gkx{cntr}); 
    cntr = cntr + 1 ;
    
end
hold off;
grid on;