import Plot from 'react-plotly.js';

const Figure = (props) => {
  return (
    <Plot
      data={[
        {
          x: props.index.length ? props.index : [...Array(props.value.length).keys()],
          y: [...props.value],
          type: 'scatter',
          mode: props.mode,
          marker: {color: 'red'},
        }
      ]}
      layout={ {width: window.screen.width * 0.9, height: 240, title: props.title} }
    />
  )
}

const MixedFigure = (props) => {

  const extractInfo = (data_) => {

    let maxValuePositive = 0
    let minValuePositive = Infinity
    let maxValueNegative = 0
    let minValueNegative = Infinity
    data_.value.forEach(v => {
      if (v > 0) {
        if (v > maxValuePositive) maxValuePositive = v
        if (v < minValuePositive) minValuePositive = v
      } else {
        if (v > maxValueNegative) maxValueNegative = v
        if (v < minValueNegative) minValueNegative = v       
      }
    })

    return {
      x: data_.index.length ? data_.index : [...Array(data_.value.length).keys()],
      y: data_.index.length ? data_.value.map(
        (v) => v > 0 ?
          (v - minValuePositive) / (maxValuePositive - minValuePositive) * 2.0:
          (v - minValueNegative) / (maxValueNegative - minValueNegative) * 2.0 - 4.0
      ) : data_.value,
      type: "scatter",
      mode: data_.type,
      name: data_.name,
      marker: {
        size: 3
      },
      line: {
        width: 1
      }    
    }
  }

  return (
    <Plot
      data={props.data.map(extractInfo)}
      layout={ {width: window.screen.width * 0.99, height: 480, title: "Summary", legend: { x: 0.0, y: 0.0 }} }
    />
  )
}

export {
  Figure,
  MixedFigure
}