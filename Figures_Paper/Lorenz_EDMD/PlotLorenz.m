%% Plot skript Lorenz Paper QuasiModo

flagNonlinCon = true;
flagContRef = true;

string = "Lorenz_EDMD_";

if flagNonlinCon
    string = string + "NonlinU_";
else
    string = string + "LinU_";
end

if flagContRef
    string = string + "RefCont";
else
    string = string + "RefPWC";
end

% load data
sol_cont = load(string + "_cont.mat");
sol_SUR = load(string + "_SUR.mat");

% plot solution

figure;
set(gca,'FontSize',18)
subplot(3,1,1); hold on;
plot(t,sol_SUR.z(:,2),'Linewidth',1.5)
plot(t,sol_cont.z(:,2),'Linewidth',1.5)
if flagContRef
    plot(t,1.5*sin(4 * pi * t / 20),'k--','Linewidth',1.5)
end
xlim([0,20.0])
ylabel("$y_2$", 'Interpreter', 'latex','FontSize', 20)

subplot(3,1,2); hold on;
plot(t,sol_SUR.J,'Linewidth',1.5)
plot(t,sol_cont.J,'Linewidth',1.5)
xlim([0,20.0])
ylabel("$J$", 'Interpreter', 'latex','FontSize', 20)

subplot(3,1,3); hold on;
plot(t,sol_SUR.u,'Linewidth',1.5)
plot(t,sol_cont.u,'Linewidth',1.5)
xlim([0,20.0])
ylim([min(min(sol_cont.u),min(sol_SUR.u)),max(max(sol_cont.u),max(sol_SUR.u))])
ylabel("$u$", 'Interpreter', 'latex','FontSize', 20)
xlabel("$t$", 'Interpreter', 'latex','FontSize', 20)


export_fig(string,'-pdf', '-r500', '-transparent');