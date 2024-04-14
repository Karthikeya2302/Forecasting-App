        $(document).ready(function() {
            $('#filter_form').submit(function(e) {
                e.preventDefault();
                var selectedSpeciality = $('#speciality_code').val();
                var selectedProcedure =  $('#procedure_id').val();
                var selectedFrequency =  $('#forecast_horizon').val();
                $.ajax({
                    url: '/predict_arima',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({'specialty': selectedSpeciality, 'procedure': selectedProcedure, 'frequency': selectedFrequency }),
                    success: function(response) 
                    {
                        displayForecast(response.forecast);
                    }
                });

                $.ajax({
                    url: '/predict_holt_winter',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({'specialty': selectedSpeciality, 'procedure': selectedProcedure, 'frequency': selectedFrequency }),
                    success: function(response) 
                    {
                        displayForecast(response.forecast);
                    }
                });

                $.ajax({
                    url: '/predict_simple_exponential_smoothing',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({'specialty': selectedSpeciality, 'procedure': selectedProcedure, 'frequency': selectedFrequency }),
                    success: function(response) 
                    {
                        displayForecast(response.forecast);
                    }
                });


                $.ajax({
                    url: '/predict_moving_average',
                    type: 'POST',
                    contentType: 'application/json',
                    data: JSON.stringify({'specialty': selectedSpeciality, 'procedure': selectedProcedure, 'frequency': selectedFrequency }),
                    success: function(response) 
                    {
                        displayForecast(response.forecast);
                    }
                });




            })

        });

          

    