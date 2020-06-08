const canvas = d3.select(".canva").attr('height', 300)
                                  .attr('width', 300)
 
// create svg canvas
const svg = canvas.append("svg").attr("width", 800)
                                .attr("height", 200);

/* svg.append("circle").attr("cx", 30)
                    .attr("cy", 30)
                    .attr("r", 15)
                    .attr("fill", "blue");

svg.append("rect").attr("x", 140)
                  .attr("y", 20)
                  .attr("height", 120)
                  .attr("width", 20)
                  .attr("rx", 50)
                  .attr("ry", 30);

svg.append("line").attr("x1", 20)
                  .attr("y1", 20)
                  .attr("x2", 60)
                  .attr("y2", 80)
                  .attr("stroke", "gray");
                  
svg.append("text").text("Hola")
                  .attr('x', 200)
                  .attr('y', 100)
                  .attr('text-anchor', 'begin')
                  .attr('stroke', 'black')
                  .attr('fill', 'orange')
                  .attr('font-size', 40);

svg.append("text").text("Hola")
                  .attr('x', 100)
                  .attr('y', 140)
                  .attr('text-anchor', 'middle')
                  .attr('fill', 'orange')
                  .attr('font-size', 40);

svg.append("text").text("Hola")
                  .attr('x', 100)
                  .attr('y', 180)
                  .attr('text-anchor', 'end')
                  .attr('fill', 'orange')
                  .attr('font-size', 40);