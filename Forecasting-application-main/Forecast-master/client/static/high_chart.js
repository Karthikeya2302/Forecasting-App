
$(document).ready(function() {
    // $('#myTable').css("display", 'none')
    $('#filter_form').submit(function(e) {
        e.preventDefault();
        $('#loader').show();
        var selectedSpeciality = $('#speciality_code').val();
        var selectedProcedure =  $('#procedure_id').val();
        var selectedFrequency =  $('#forecast_horizon').val();
        var selectedModel = $('#select_model').val();
        if (selectedModel === 'arima') {
            $.ajax({
                url: '/main/predict_arima',
                type: 'POST',
                dataType:'json',
                contentType: 'application/json',
                data: JSON.stringify({'specialty': selectedSpeciality, 'procedure': selectedProcedure, 'frequency': selectedFrequency }),
                success: function(response) {
                    $('#myTable').css("display","table")
                    bindGrid(response,selectedSpeciality,selectedProcedure)
                    var dates = response.dates;
                    var actual = response.actual;
                    var name = response.name;
                    var forecasted_values = response.forecast_values;
                    var forecasted_size = response.forecast_size;
                    var lower_ci=response.lower_ci;
                    var upper_ci=response.upper_ci;

                    Highcharts.chart('chartContainer', {
                      chart: {
                        style: {
                          fontFamily: 'Arial, sans-serif',
                          backgroundColor: '#f5f5f5',
                        },
                        zoomType: 'x', // Enable zooming on the x-axis
                      },
                      title: {
                        text: 'Procedural Forecasting',
                        align: 'center',
                        style: {
                          fontSize: '18px',
                          fontWeight: 'bold',
                        },
                      },
                      yAxis: {
                        title: {
                          text: 'Number of Procedures',
                          style: {
                            fontWeight: 'bold',
                          },
                        },
                        labels: {
                          style: {
                            fontSize: '12px',
                          },
                        },
                      },
                      xAxis: {
                        categories: dates,
                        labels: {
                          style: {
                            fontSize: '12px',
                          },
                        },
                      },
                      series: [
                        {
                          name: name,
                          data: forecasted_values,
                          zoneAxis: 'x',
                          zones: [
                            {
                              value: forecasted_values.length - forecasted_size,
                              color: '#0000ff',
                            },
                            {
                              value: forecasted_values.length,
                              color: '#ff0000',
                            },
                            {
                              dashStyle: 'dot',
                              color: '#00ff00',
                            },
                          ],
                          lineWidth: 2,
                          marker: {
                            enabled: true,
                            radius: 4,
                          },
                        },
                        {
                          name: 'Actual',
                          data: actual,
                          color: 'orange',
                          lineWidth: 2,
                          marker: {
                            enabled: true,
                            radius: 4,
                          },
                        },
                        {
                          name: 'Lower',
                          data: lower_ci,
                          color: 'blue',
                          lineWidth: 1,
                          dashStyle: 'dash',
                          marker: {
                            enabled: false,
                          },
                        },
                        {
                          name: 'Upper',
                          data: upper_ci,
                          color: 'blue',
                          lineWidth: 1,
                          dashStyle: 'dash',
                          marker: {
                            enabled: false,
                          },
                        },
                      ],
                    
                        legend: {
                          itemStyle: {
                            fontSize: '12px'                 // Adjust the font size of the legend items
                          }
                        }
                      });
                      
                    $('#loader').hide();
                }
                
            });
        } 
         else if (selectedModel === 'ses') {
            $.ajax({
                url: '/main/predict_simple_exponential_smoothing',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({'specialty': selectedSpeciality, 'procedure': selectedProcedure, 'frequency': selectedFrequency }),
                success: function(response) {
                    //$('#myTable').css("display","initial")
                    bindGrid(response,selectedSpeciality,selectedProcedure)
                    var dates = response.dates;
                    var actual = response.actual;
                    var name = response.name;
                    var forecasted_values = response.forecast_values;
                    var forecasted_size = response.forecast_size;
                    var lower_ci=response.lower_ci;
                    var upper_ci=response.upper_ci;

                    Highcharts.chart('chartContainer', {
                      chart: {
                        style: {
                          fontFamily: 'Arial, sans-serif',
                          backgroundColor: '#f5f5f5',
                        },
                        zoomType: 'x', // Enable zooming on the x-axis
                      },
                      title: {
                        text: 'Procedural Forecasting',
                        align: 'center',
                        style: {
                          fontSize: '18px',
                          fontWeight: 'bold',
                        },
                      },
                      yAxis: {
                        title: {
                          text: 'Number of Procedures',
                          style: {
                            fontWeight: 'bold',
                          },
                        },
                        labels: {
                          style: {
                            fontSize: '12px',
                          },
                        },
                      },
                      xAxis: {
                        categories: dates,
                        labels: {
                          style: {
                            fontSize: '12px',
                          },
                        },
                      },
                      series: [
                        {
                          name: name,
                          data: forecasted_values,
                          zoneAxis: 'x',
                          zones: [
                            {
                              value: forecasted_values.length - forecasted_size,
                              color: '#0000ff',
                            },
                            {
                              value: forecasted_values.length,
                              color: '#ff0000',
                            },
                            {
                              dashStyle: 'dot',
                              color: '#00ff00',
                            },
                          ],
                          lineWidth: 2,
                          marker: {
                            enabled: true,
                            radius: 4,
                          },
                        },
                        {
                          name: 'Actual',
                          data: actual,
                          color: 'orange',
                          lineWidth: 2,
                          marker: {
                            enabled: true,
                            radius: 4,
                          },
                        },
                        {
                          name: 'Lower',
                          data: lower_ci,
                          color: 'blue',
                          lineWidth: 1,
                          dashStyle: 'dash',
                          marker: {
                            enabled: false,
                          },
                        },
                        {
                          name: 'Upper',
                          data: upper_ci,
                          color: 'blue',
                          lineWidth: 1,
                          dashStyle: 'dash',
                          marker: {
                            enabled: false,
                          },
                        },
                      ],
                        legend: {
                          itemStyle: {
                            fontSize: '12px'                 // Adjust the font size of the legend items
                          }
                        }
                      });
                      
                    $('#loader').hide();
                }
                
            });
        } else if (selectedModel === 'hw') {
            $.ajax({
                url: '/main/predict_holt_winter',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({'specialty': selectedSpeciality, 'procedure': selectedProcedure, 'frequency': selectedFrequency }),
                success: function(response) {
                    //$('#myTable').css("display","initial")
                    bindGrid(response,selectedSpeciality,selectedProcedure)
                    var dates = response.dates;
                    var actual = response.actual;
                    var name = response.name;
                    var forecasted_values = response.forecast_values;
                    var forecasted_size = response.forecast_size;
                    var lower_ci=response.lower_ci;
                    var upper_ci=response.upper_ci;

                    Highcharts.chart('chartContainer', {
                      chart: {
                        style: {
                          fontFamily: 'Arial, sans-serif',
                          backgroundColor: '#f5f5f5',
                        },
                        zoomType: 'x', // Enable zooming on the x-axis
                      },
                      title: {
                        text: 'Procedural Forecasting',
                        align: 'center',
                        style: {
                          fontSize: '18px',
                          fontWeight: 'bold',
                        },
                      },
                      yAxis: {
                        title: {
                          text: 'Number of Procedures',
                          style: {
                            fontWeight: 'bold',
                          },
                        },
                        labels: {
                          style: {
                            fontSize: '12px',
                          },
                        },
                      },
                      xAxis: {
                        categories: dates,
                        labels: {
                          style: {
                            fontSize: '12px',
                          },
                        },
                      },
                      series: [
                        {
                          name: name,
                          data: forecasted_values,
                          zoneAxis: 'x',
                          zones: [
                            {
                              value: forecasted_values.length - forecasted_size,
                              color: '#0000ff',
                            },
                            {
                              value: forecasted_values.length,
                              color: '#ff0000',
                            },
                            {
                              dashStyle: 'dot',
                              color: '#00ff00',
                            },
                          ],
                          lineWidth: 2,
                          marker: {
                            enabled: true,
                            radius: 4,
                          },
                        },
                        {
                          name: 'Actual',
                          data: actual,
                          color: 'orange',
                          lineWidth: 2,
                          marker: {
                            enabled: true,
                            radius: 4,
                          },
                        },
                        {
                          name: 'Lower',
                          data: lower_ci,
                          color: 'blue',
                          lineWidth: 1,
                          dashStyle: 'dash',
                          marker: {
                            enabled: false,
                          },
                        },
                        {
                          name: 'Upper',
                          data: upper_ci,
                          color: 'blue',
                          lineWidth: 1,
                          dashStyle: 'dash',
                          marker: {
                            enabled: false,
                          },
                        },
                      ],
                        legend: {
                          itemStyle: {
                            fontSize: '12px'                 // Adjust the font size of the legend items
                          }
                        }
                      });
                      
                    $('#loader').hide();
                }
                
            });
        } else if (selectedModel === 'ma') {
            $.ajax({
                url: '/main/predict_moving_average',
                type: 'POST',
                contentType: 'application/json',
                data: JSON.stringify({'specialty': selectedSpeciality, 'procedure': selectedProcedure, 'frequency': selectedFrequency }),
                success: function(response) {
                    //$('#myTable').css("display","initial")
                    bindGrid(response,selectedSpeciality,selectedProcedure)
                    var dates = response.dates;
                    var actual = response.actual;
                    var name = response.name;
                    var forecasted_values = response.forecast_values;
                    var forecasted_size = response.forecast_size;
                    var lower_ci=response.lower_ci;
                    var upper_ci=response.upper_ci;

                    Highcharts.chart('chartContainer', {
                      chart: {
                        style: {
                          fontFamily: 'Arial, sans-serif',
                          backgroundColor: '#f5f5f5',
                        },
                        zoomType: 'x', // Enable zooming on the x-axis
                      },
                      title: {
                        text: 'Procedural Forecasting',
                        align: 'center',
                        style: {
                          fontSize: '18px',
                          fontWeight: 'bold',
                        },
                      },
                      yAxis: {
                        title: {
                          text: 'Number of Procedures',
                          style: {
                            fontWeight: 'bold',
                          },
                        },
                        labels: {
                          style: {
                            fontSize: '12px',
                          },
                        },
                      },
                      xAxis: {
                        categories: dates,
                        labels: {
                          style: {
                            fontSize: '12px',
                          },
                        },
                      },
                      series: [
                        {
                          name: name,
                          data: forecasted_values,
                          zoneAxis: 'x',
                          zones: [
                            {
                              value: forecasted_values.length - forecasted_size,
                              color: '#0000ff',
                            },
                            {
                              value: forecasted_values.length,
                              color: '#ff0000',
                            },
                            {
                              dashStyle: 'dot',
                              color: '#00ff00',
                            },
                          ],
                          lineWidth: 2,
                          marker: {
                            enabled: true,
                            radius: 4,
                          },
                        },
                        {
                          name: 'Actual',
                          data: actual,
                          color: 'orange',
                          lineWidth: 2,
                          marker: {
                            enabled: true,
                            radius: 4,
                          },
                        },
                        {
                          name: 'Lower',
                          data: lower_ci,
                          color: 'blue',
                          lineWidth: 1,
                          dashStyle: 'dash',
                          marker: {
                            enabled: false,
                          },
                        },
                        {
                          name: 'Upper',
                          data: upper_ci,
                          color: 'blue',
                          lineWidth: 1,
                          dashStyle: 'dash',
                          marker: {
                            enabled: false,
                          },
                        },
                      ],
                        legend: {
                          itemStyle: {
                            fontSize: '12px'                 // Adjust the font size of the legend items
                          }
                        }
                    });
                      
                    $('#loader').hide();
                }
                
            });
        }
    });
});

var form = document.getElementById('filter_form');
form.addEventListener('submit', function(event) {
  if (!validateForm()) {
    alert('Filling each field is mandatory.');
    event.preventDefault();
  }
});
function validateForm() {
  var specialityCode = document.getElementById('speciality_code').value;
  var procedureId = document.getElementById('procedure_id').value;
  var forecastHorizon = document.getElementById('forecast_horizon').value;
  var selectModel = document.getElementById('select_model').value;
  if (
    specialityCode === 'Select Speciality Code' ||
    procedureId === 'Select Procedure Id' ||
    forecastHorizon === 'Select Forecast Horizon' ||
    selectModel === 'Select Model'
  ) {
    return false;
  }
  return true;
}
