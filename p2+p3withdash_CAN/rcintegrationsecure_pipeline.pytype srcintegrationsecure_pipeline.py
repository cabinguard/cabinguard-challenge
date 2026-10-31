warning: in the working copy of 'src/cabinguard/pipeline.py', LF will be replaced by CRLF the next time Git touches it
[1mdiff --git a/src/cabinguard/pipeline.py b/src/cabinguard/pipeline.py[m
[1mindex f120ca0..1ec1f04 100644[m
[1m--- a/src/cabinguard/pipeline.py[m
[1m+++ b/src/cabinguard/pipeline.py[m
[36m@@ -13,6 +13,7 @@[m [mclass CabinGuardPipeline:[m
         self.guardian = SafetyGuardian(config_path)[m
         self.vehicle = SimulatedVehicle()[m
         self.secure_adapter = secure_adapter[m
[32m+[m[32m        self.secure_coordinator = None[m
         self.temporal = TemporalAnalyzer(window_seconds=5.0, min_persistence_seconds=2.0)[m
         self.digital_twin = DriverDigitalTwin(min_calibration_samples=10)[m
         self.health_manager = SystemSensorHealthManager()[m
[36m@@ -55,8 +56,29 @@[m [mclass CabinGuardPipeline:[m
         vehicle_context = self.vehicle.context[m
         if sensors.vehicle_interface_available is not True:[m
             vehicle_context.vehicle_available = False[m
[31m-        decision = self.guardian.evaluate(effective_detection, sensors, security, vehicle_context)[m
[31m-        record = self.vehicle.apply_decision(decision)[m
[32m+[m
[32m+[m[32m        decision = self.guardian.evaluate([m
[32m+[m[32m            effective_detection, sensors, security, vehicle_context[m
[32m+[m[32m        )[m
[32m+[m
[32m+[m[32m        # Default to a safe, logging-only decision.[m
[32m+[m[32m        safe_decision = replace([m
[32m+[m[32m            decision,[m
[32m+[m[32m            allow_simulated_response=False,[m
[32m+[m[32m            actions=["LOG_EVENT"],[m
[32m+[m[32m        )[m
[32m+[m
[32m+[m[32m        # Critical simulated responses require the P3 coordinator.[m
[32m+[m[32m        if self.secure_coordinator is not None:[m
[32m+[m[32m            try:[m
[32m+[m[32m                safe_decision = self.secure_coordinator.process_decision([m
[32m+[m[32m                    decision,[m
[32m+[m[32m                    now=detection.timestamp,[m
[32m+[m[32m                )[m
[32m+[m[32m            except Exception:[m
[32m+[m[32m                pass  # Fail closed: keep the logging-only decision.[m
[32m+[m
[32m+[m[32m        record = self.vehicle.apply_decision(safe_decision)[m
         record.update({[m
             "degraded_mode": decision.degraded_mode or health.system_health_state != "FULL",[m
             "rejected_checks": decision.rejected_checks,[m
