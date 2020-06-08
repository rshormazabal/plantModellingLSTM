const canvas = d3.select('.canva')

canvas.attr('height', 300)
      .attr('width', 400);

const svg = canvas.append('svg').attr('height', 300)
                                .attr('width', 400);

                                
const rect = svg.selectAll('rect');

d3.json('text.json')
    .then(data => {
        rect.data(data)
            .enter().append('rect')
            .attr('x', (d, i) => i*50)
            .attr('y', d => 300 - d.height - 5)
            .attr('width', 49)
            .attr('height', d => d.height + 10)
            .attr('fill', d => d.fill)
            .attr('rx', 5);
    })
