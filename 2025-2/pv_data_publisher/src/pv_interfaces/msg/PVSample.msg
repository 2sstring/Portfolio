# PV 설비 1개 인버터의 1샘플 데이터
builtin_interfaces/Time stamp	# 타임스템프(ROS2 표준 시간 타입)
string plant_id			# 발전소 ID
string inverter_id		# 인버터 ID
int32 hour			# Hour
float32 dhi			# Diffuse Horizontal Irradiance [W/m^2]
float32 dni			# Direct Normal Irradiance [W/m^2]
float32 module_temp		# Module Temperature [℃]
float32 ambient_temp		# Temperature [℃]
float32 ws			# 풍속 [m/s]
float32 rh			# 상대습도 [%]
float32 ac_power		# Power [W]
string status			# "OK", "TRIP", "LOW_OUTPUT"
